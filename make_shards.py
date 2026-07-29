"""make_shards.py — генерация обучающих шардов из tablebase.

Формат записи — 14 байт uint8:
  [0:10] лунки (0..50)   [10] K1//2   [11] K2//2
  [12]   WDL (0=loss, 1=draw, 2=win)
  [13]   битовая маска оптимальных ходов (биты 0..4)

Проверка порта rank/unrank БЕЗ tablebase (бесплатно, локально):
  python make_shards.py --selftest
Самопроверка ДО генерации (обязательна, см. раздел 4.4):
  python make_shards.py --tb /workspace/tablebase --verify 2000
Полный запуск:
  python make_shards.py --tb /workspace/tablebase --out /workspace/shards \
      --n 500_000_000 --workers 24
"""
import argparse, os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from bestemshe_core import (Tablebase, KAZANS, sample_position, layer_weights,
                            count_states, rank, unrank, legal_moves, apply_move,
                            verify_consistency, STONES)

BLOCK = 50_000   # позиций одного слоя подряд — бережём кэш распакованных слоёв


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb")              # обязателен для всего, кроме --selftest
    ap.add_argument("--out", default="/workspace/shards")
    ap.add_argument("--n", type=int, default=500_000_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shard-size", type=int, default=20_000_000)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
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

    os.makedirs(a.out, exist_ok=True)
    per_worker = a.shard_size // a.workers
    n_shards = a.n // a.shard_size
    with Pool(a.workers) as pool:
        for s in tqdm(range(n_shards), desc="шарды", unit="шард"):
            jobs = [(s * a.workers + w, per_worker, a.tb)
                    for w in range(a.workers)]
            # imap сохраняет порядок jobs; бар — по завершённым воркерам шарда
            shard = np.concatenate(list(tqdm(
                pool.imap(_worker, jobs), total=len(jobs),
                desc=f"shard {s:04d}", unit="воркер", leave=False)))
            shard.tofile(os.path.join(a.out, f"shard_{s:04d}.bin"))
            tqdm.write(f"shard {s + 1}/{n_shards} готов"
                       f" ({(s + 1) * a.shard_size:,} позиций)")


if __name__ == "__main__":
    main()