"""bestemshe_core.py — ядро: кодирование позиций, ранжирование, доступ к tablebase.

ВНИМАНИЕ: legal_moves()/apply_move() записаны по правилам тогызкумалака,
адаптированным к 5 лункам. Они ОБЯЗАНЫ бит-в-бит совпадать с решателем
github.com/ansarzeinulla/Bestemshe. Перед использованием замените их кодом
из вашего репозитория и прогоните: python make_shards.py --tb ... --verify 2000
То же касается порядка ранжирования rank()/unrank().
"""
import functools, os
from collections import OrderedDict
from math import comb as _comb
import numpy as np
import zstandard as zstd

N_PITS = 10      # 10 лунок (5 своих + 5 соперника)
STONES = 50      # инвариант: sum(pits) + K1 + K2 == 50
KAZANS = [(k1, k2) for k1 in range(0, 26, 2) for k2 in range(0, 26, 2)]  # 169 слоёв


@functools.lru_cache(maxsize=None)
def comb(n, k):
    return _comb(n, k) if n >= 0 and 0 <= k <= n else 0


def count_states(total, pits=N_PITS):
    """Число раскладок total камней по pits лункам (stars and bars)."""
    return comb(total + pits - 1, pits - 1)


def rank(pits_cfg):
    """Конфигурация лунок -> StateIndex. Порядок сверить с решателем!"""
    idx, rest, left = 0, sum(pits_cfg), len(pits_cfg)
    for v in pits_cfg:
        for w in range(v):
            idx += comb(rest - w + left - 2, left - 2)
        rest -= v
        left -= 1
    return idx


def unrank(idx, total, pits=N_PITS):
    """StateIndex -> конфигурация лунок (обратная к rank)."""
    cfg, rest, left = [], total, pits
    for _ in range(pits - 1):
        v = 0
        while True:
            block = comb(rest - v + left - 2, left - 2)
            if idx < block:
                break
            idx -= block
            v += 1
        cfg.append(v); rest -= v; left -= 1
    cfg.append(rest)
    return cfg


@functools.lru_cache(maxsize=1)
def _weights():
    w = [count_states(STONES - k1 - k2) for k1, k2 in KAZANS]
    s = float(sum(w))
    return tuple(x / s for x in w)


def layer_weights():
    return np.array(_weights())


def sample_position(rng):
    """Слой — пропорционально числу состояний, индекс внутри слоя — равномерно."""
    k1, k2 = KAZANS[int(rng.choice(len(KAZANS), p=layer_weights()))]
    total = STONES - k1 - k2
    return (k1, k2, unrank(int(rng.integers(count_states(total))), total))


class Tablebase:
    """Доступ к битсетам win/draw. Держит в RAM до max_layers распакованных
    слоёв (крупнейший, (0,0)_win, ~1.6 ГБ в развёрнутом виде)."""

    def __init__(self, root, max_layers=8):
        self.root, self.max_layers = root, max_layers
        self._cache = OrderedDict()

    def _bits(self, k1, k2, kind):
        key = (k1, k2, kind)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = os.path.join(self.root, "data", f"layer_{k1}_{k2}_{kind}.bin.zst")
        with open(path, "rb") as fh:
            data = zstd.ZstdDecompressor().decompress(fh.read())
        self._cache[key] = data
        if len(self._cache) > self.max_layers:
            self._cache.popitem(last=False)
        return data

    def value(self, pos):
        """0 = LOSS, 1 = DRAW, 2 = WIN для стороны на ходу."""
        k1, k2, pits = pos
        if k1 >= 26: return 2          # казан >= 26 решает игру немедленно
        if k2 >= 26: return 0
        idx = rank(tuple(pits))
        w = self._bits(k1, k2, "win")
        if (w[idx >> 3] >> (idx & 7)) & 1: return 2   # бит-адресация как в
        d = self._bits(k1, k2, "draw")                # Quickstart датасета
        if (d[idx >> 3] >> (idx & 7)) & 1: return 1
        return 0


def legal_moves(pos):
    """Свои лунки — индексы 0..4; ход возможен из непустой лунки."""
    return [i for i in range(5) if pos[2][i] > 0]


def apply_move(pos, move):
    """Ход стороны на ходу. Возвращает позицию С ТОЧКИ ЗРЕНИЯ СОПЕРНИКА
    (канонический вид tablebase). <<< ЗАМЕНИТЕ ТЕЛО КОДОМ ВАШЕГО РЕШАТЕЛЯ >>>"""
    k1, k2, pits = pos
    pits = list(pits)
    stones = pits[move]
    if stones == 1:                      # одиночный камень — в следующую лунку
        pits[move] = 0
        last = (move + 1) % N_PITS
        pits[last] += 1
    else:                                # оставить 1, остальные посеять дальше
        pits[move] = 1
        cur = move
        for _ in range(stones - 1):
            cur = (cur + 1) % N_PITS
            pits[cur] += 1
        last = cur
    if 5 <= last <= 9 and pits[last] % 2 == 0:   # захват: чётная лунка соперника
        k1 += pits[last]
        pits[last] = 0
    return (k2, k1, pits[5:] + pits[:5])         # переворот доски


def verify_consistency(tb, n=1000, seed=0):
    """Теоретико-игровая самопроверка: V(pos) == max по ходам (2 - V(child)).
    Ловит ЛЮБОЕ расхождение правил или индексации с решателем до обучения."""
    rng = np.random.default_rng(seed)
    bad = checked = 0
    for _ in range(n):
        pos = sample_position(rng)
        moves = legal_moves(pos)
        if not moves:
            continue                     # терминальные правила сверьте отдельно
        checked += 1
        best = max(2 - tb.value(apply_move(pos, m)) for m in moves)
        if tb.value(pos) != best:
            bad += 1
    print(f"consistency: {checked - bad}/{checked} ok")
    return bad == 0