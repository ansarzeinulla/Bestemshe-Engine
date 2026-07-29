"""make_shards.py — генерация обучающих шардов из tablebase.

Формат записи — 14 байт uint8:
  [0:10] лунки (0..50)   [10] K1//2   [11] K2//2
  [12]   WDL (0=loss, 1=draw, 2=win)
  [13]   битовая маска оптимальных ходов (биты 0..4)

Самопроверка ДО генерации (обязательна, см. раздел 4.4):
  python make_shards.py --tb /workspace/tablebase --verify 2000
Полный запуск:
  python make_shards.py --tb /workspace/tablebase --out /workspace/shards \
      --n 500_000_000 --workers 24
"""
import argparse, os
import numpy as np
from multiprocessing import Pool
from bestemshe_core import (Tablebase, KAZANS, sample_position, layer_weights,
                            count_states, unrank, legal_moves, apply_move,
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
    ap.add_argument("--tb", required=True)
    ap.add_argument("--out", default="/workspace/shards")
    ap.add_argument("--n", type=int, default=500_000_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shard-size", type=int, default=20_000_000)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()

    if a.verify:                         # режим самопроверки — без генерации
        ok = verify_consistency(Tablebase(a.tb), n=a.verify)
        raise SystemExit(0 if ok else
                         "СТОП: правила/индексация НЕ совпадают с tablebase")

    os.makedirs(a.out, exist_ok=True)
    per_worker = a.shard_size // a.workers
    n_shards = a.n // a.shard_size
    with Pool(a.workers) as pool:
        for s in range(n_shards):
            jobs = [(s * a.workers + w, per_worker, a.tb)
                    for w in range(a.workers)]
            shard = np.concatenate(pool.map(_worker, jobs))
            shard.tofile(os.path.join(a.out, f"shard_{s:04d}.bin"))
            print(f"shard {s + 1}/{n_shards} готов"
                  f" ({(s + 1) * a.shard_size:,} позиций)")


if __name__ == "__main__":
    main()