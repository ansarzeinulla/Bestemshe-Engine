"""make_shards.py — генерация обучающих шардов из tablebase.

Формат записи — 14 байт uint8:
  [0:10] лунки (0..50)   [10] K1//2   [11] K2//2
  [12]   WDL (0=loss, 1=draw, 2=win)
  [13]   битовая маска оптимальных ходов (биты 0..4)

Проверка порта rank/unrank БЕЗ tablebase (бесплатно, локально):
  python make_shards.py --selftest
Самопроверка ДО генерации (обязательна, см. раздел 4.4):
  python make_shards.py --tb /workspace/tablebase --verify 2000
Полный запуск (режим обязателен: overwrite или append):
  python make_shards.py --tb /workspace/tablebase --out /workspace/shards \
      --n 500_000_000 --workers 24 --mode overwrite

ИЗМЕНЕНИЯ (continuous learning):
  --mode overwrite|append  — overwrite: удаляет старые shard_*.bin перед стартом
                             и нумерует с 0; append: продолжает нумерацию с
                             последнего существующего шарда, старые файлы не трогает.
  generation_status.json  — пишется в --out атомарно (через os.replace) после
                             каждого завершённого воркера, чтобы `cat
                             generation_status.json` в соседнем терминале всегда
                             показывал консистентный прогресс/скорость/ETA.
"""
import argparse, datetime, glob, json, os, re, time
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from bestemshe_core import (Tablebase, KAZANS, sample_position, layer_weights,
                            count_states, rank, unrank, legal_moves, apply_move,
                            verify_consistency, STONES)

BLOCK = 50_000   # позиций одного слоя подряд — бережём кэш распакованных слоёв
STATUS_FILE = "generation_status.json"


def optimal_move_mask(tb, pos):
    """Ход оптимален, если даёт max(2 - V(child)) — лучший исход для ходящего."""
    moves = legal_moves(pos)
    if not moves:
        return 0
    child = {m: tb.value(apply_move(pos, m)) for m in moves}
    best = max(2 - v for v in child.values())
    return sum(1 << m for m, v in child.items() if 2 - v == best)


def _compositions(total, parts):
    """Все раскладки total камней по parts лункам (для полного перебора)."""
    if parts == 1:
        yield (total,)
        return
    for v in range(total + 1):
        for rest in _compositions(total - v, parts - 1):
            yield (v,) + rest


def selftest():
    """Проверка соответствия rank/unrank индексации решателя (StateIndex.h /
    Solver.h::IndexBoard) без tablebase. Якорные значения посчитаны C++ кодом.
    ВАЖНО: при R=1 колекс- и лекс-порядки совпадают, поэтому тестируем R>=2."""
    # Якорь из C++ IndexBoard: доска (0,2,0,...,0) -> индекс 52 (лекс-rank дал бы 44)
    anchor = (0, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    assert rank(anchor) == 52, f"rank{anchor} = {rank(anchor)}, ожидается 52 (C++ IndexBoard)"
    assert unrank(52, 2) == list(anchor), f"unrank(52, 2) = {unrank(52, 2)}, ожидается {list(anchor)}"

    for R in range(2, 6):                # полный перебор: биекция + round-trip
        n = count_states(R)
        seen = set()
        for cfg in tqdm(_compositions(R, 10), total=n,
                        desc=f"selftest R={R}", unit="сост", leave=False):
            r = rank(cfg)
            assert 0 <= r < n and r not in seen, f"rank не биекция на R={R}: {cfg} -> {r}"
            seen.add(r)
            assert unrank(r, R) == list(cfg), f"round-trip провален: {cfg} -> {r} -> {unrank(r, R)}"
        assert len(seen) == n
        # unrank(0) = (0,...,0,R): согласовано с wrap в Solver.cpp (AdvanceBoard)
        assert unrank(0, R) == [0] * 9 + [R]
        print(f"selftest R={R}: {n} состояний, биекция и round-trip ok")

    rng = np.random.default_rng(0)       # случайные round-trip на полном диапазоне
    for _ in tqdm(range(2000), desc="round-trip", unit="поз", leave=False):
        R = int(rng.integers(0, 51))
        i = int(rng.integers(count_states(R)))
        assert rank(tuple(unrank(i, R))) == i, f"round-trip провален: R={R}, idx={i}"
    print("selftest: 2000 случайных round-trip (R до 50) ok")
    print("selftest: OK — rank/unrank совпадают с индексацией решателя")


def _worker(args):
    seed, count, tb_root = args
    rng = np.random.default_rng(seed)
    tb = Tablebase(tb_root)
    out = np.empty((count, 14), dtype=np.uint8)
    i = 0
    while i < count:                     # блоками по одному слою (K1, K2)
        k1, k2 = KAZANS[int(rng.choice(len(KAZANS), p=layer_weights()))]
        n_states = count_states(STONES - k1 - k2)
        for _ in range(min(BLOCK, count - i)):
            pits = unrank(int(rng.integers(n_states)), STONES - k1 - k2)
            pos = (k1, k2, pits)
            out[i, :10] = pits
            out[i, 10], out[i, 11] = k1 // 2, k2 // 2
            out[i, 12] = tb.value(pos)
            out[i, 13] = optimal_move_mask(tb, pos)
            i += 1
    return out


_SHARD_RE = re.compile(r"shard_(\d+)\.bin$")


def _existing_shards(out_dir):
    """Отсортированный список (индекс, путь) уже лежащих в out_dir шардов."""
    found = []
    for p in glob.glob(os.path.join(out_dir, "shard_*.bin")):
        m = _SHARD_RE.search(p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


class StatusWriter:
    """Атомарно пишет generation_status.json — можно `cat` в любой момент
    и всегда получать консистентный JSON (пишем во временный файл + os.replace)."""

    def __init__(self, path, mode, out_dir, target_n, existing_positions, existing_shards):
        self.path = path
        self.mode = mode
        self.out_dir = out_dir
        self.target_n = target_n
        self.existing_positions = existing_positions
        self.existing_shards = existing_shards
        self.generated = 0
        self.shards_written = 0
        self.t0 = time.time()

    def update(self, generated=None, shards_written=None, status="running", extra=None):
        if generated is not None:
            self.generated = generated
        if shards_written is not None:
            self.shards_written = shards_written
        elapsed = time.time() - self.t0
        speed = self.generated / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.target_n - self.generated)
        eta_sec = remaining / speed if speed > 0 else None
        payload = {
            "status": status,
            "mode": self.mode,
            "out_dir": self.out_dir,
            "generated_this_run": self.generated,
            "target_this_run": self.target_n,
            "shards_written_this_run": self.shards_written,
            "total_positions_on_disk": self.existing_positions + self.generated,
            "total_shards_on_disk": self.existing_shards + self.shards_written,
            "elapsed_sec": round(elapsed, 1),
            "speed_pos_per_sec": round(speed, 1),
            "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
            "eta_human": str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec is not None else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)  # атомарная замена — cat никогда не увидит "рваный" JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb")              # обязателен для всего, кроме --selftest
    ap.add_argument("--out", default="/workspace/shards")
    ap.add_argument("--n", type=int, default=500_000_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shard-size", type=int, default=20_000_000)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["overwrite", "append"],
                    help="overwrite: удалить старые шарды и начать с 0; "
                         "append: продолжить нумерацию с существующих файлов")
    a = ap.parse_args()

    if a.selftest:                       # проверка rank/unrank без tablebase
        selftest()
        raise SystemExit(0)

    if not a.tb:
        ap.error("--tb обязателен (кроме режима --selftest)")

    if a.verify:                         # режим самопроверки — без генерации
        ok = verify_consistency(Tablebase(a.tb), n=a.verify)
        raise SystemExit(0 if ok else
                         "СТОП: правила/индексация НЕ совпадают с tablebase")

    if not a.mode:
        ap.error("--mode обязателен: overwrite или append")

    os.makedirs(a.out, exist_ok=True)

    existing = _existing_shards(a.out)
    if a.mode == "overwrite":
        for _, p in existing:
            os.remove(p)
        start_shard = 0
        existing_positions = 0
        existing_shards = 0
    else:  # append
        start_shard = (existing[-1][0] + 1) if existing else 0
        existing_shards = len(existing)
        # размер существующих файлов в позициях (14 байт/запись) — для статуса
        existing_positions = sum(os.path.getsize(p) // 14 for _, p in existing)

    status = StatusWriter(os.path.join(a.out, STATUS_FILE), a.mode, a.out,
                          a.n, existing_positions, existing_shards)
    status.update(generated=0, shards_written=0)

    per_worker = a.shard_size // a.workers
    n_shards = a.n // a.shard_size
    generated = 0
    try:
        with Pool(a.workers) as pool:
            for s in tqdm(range(n_shards), desc="шарды", unit="шард"):
                shard_idx = start_shard + s
                jobs = [(shard_idx * a.workers + w, per_worker, a.tb)
                        for w in range(a.workers)]
                # imap сохраняет порядок jobs; бар — по завершённым воркерам шарда
                chunks = []
                for chunk in tqdm(pool.imap(_worker, jobs), total=len(jobs),
                                  desc=f"shard {shard_idx:04d}", unit="воркер", leave=False):
                    chunks.append(chunk)
                    generated += len(chunk)
                    status.update(generated=generated, shards_written=s)
                shard = np.concatenate(chunks)
                shard.tofile(os.path.join(a.out, f"shard_{shard_idx:04d}.bin"))
                status.update(generated=generated, shards_written=s + 1)
                tqdm.write(f"shard {shard_idx:04d} готов ({generated:,} позиций в этом запуске)")
    except BaseException:
        status.update(status="failed")
        raise
    else:
        status.update(status="done")


if __name__ == "__main__":
    main()
