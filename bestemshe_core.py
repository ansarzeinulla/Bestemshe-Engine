"""bestemshe_core.py — ядро: кодирование позиций, ранжирование, доступ к tablebase.

rank()/unrank() — прямой порт Solver.h::IndexBoard / StateIndex::UnindexState
(колексикографическая комбинаторная система счисления); apply_move()/
legal_moves() эквивалентны BestemsheCore.h::ExecuteMoveAndFlip; терминальные
правила (пустая сторона / казан >= 26) — как в Solver.cpp и query.cpp.
Проверка без tablebase: python make_shards.py --selftest
Проверка с tablebase:   python make_shards.py --tb ... --verify 2000
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
    """Конфигурация лунок -> индекс бита в слое.
    Порт Solver.h::IndexBoard (колекс комбинаторная система счисления):
    I_B = sum_{i=0..8} C(i + prefix_sum(board[0..i]), i+1)."""
    idx = s = 0
    for i in range(9):
        s += pits_cfg[i]
        idx += comb(i + s, i + 1)
    return idx


def unrank(idx, total, pits=N_PITS):
    """Индекс бита -> конфигурация лунок (обратная к rank).
    Порт StateIndex::UnindexState (доска-часть)."""
    cfg = [0] * pits
    current_p = 0
    for i in range(8, -1, -1):
        p = i
        while comb(p + 1, i + 1) <= idx:
            p += 1
        idx -= comb(p, i + 1)
        if i == 8:
            cfg[9] = total + 8 - p
        else:
            cfg[i + 1] = current_p - p - 1
        current_p = p
    cfg[0] = current_p
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


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"        # магия zstd-фрейма (little-endian)
BLOCK_BYTES = 33554432 // 8             # 4 МиБ — Compressor::CompressMicroLayer


def _unzstd(data, max_size):
    """Распаковка одного zstd-фрейма; работает и без content-size в заголовке."""
    dctx = zstd.ZstdDecompressor()
    try:
        return dctx.decompress(data, max_output_size=max_size)
    except zstd.ZstdError:
        return dctx.decompressobj().decompress(data)


def _parse_container(data, expected):
    """Порт Compressor::DecompressMicroLayer: контейнер
    [u32 num_blocks][u32 offsets[num_blocks+1]][независимые zstd-блоки по 4 МиБ].
    Возвращает битсет ровно expected байт или None, если это не контейнер."""
    if len(data) < 8:
        return None
    num_blocks = int.from_bytes(data[:4], "little")
    if not 0 < num_blocks <= expected // BLOCK_BYTES + 2:
        return None
    header = 4 + 4 * (num_blocks + 1)
    if len(data) < header:
        return None
    offsets = np.frombuffer(data[4:header], dtype="<u4").astype(np.int64)
    if offsets[0] != header or offsets[-1] > len(data) or np.any(np.diff(offsets) < 0):
        return None
    dctx = zstd.ZstdDecompressor()
    out = bytearray()
    for b in range(num_blocks):
        out += dctx.decompress(data[offsets[b]:offsets[b + 1]],
                               max_output_size=BLOCK_BYTES)
    return bytes(out[:expected])


class Tablebase:
    """Доступ к битсетам win/draw. Держит в RAM до max_layers распакованных
    слоёв (крупнейший, (0,0)_win, ~1.6 ГБ в развёрнутом виде)."""

    def __init__(self, root, max_layers=8):
        self.root, self.max_layers = root, max_layers
        self._cache = OrderedDict()

    def _find(self, name):
        """Файл слоя ищется в root/ и root/data/ с суффиксами
        .bin.zst (HF-датасет), .bin (layers/compressed), .raw (сырой битсет)."""
        for sub in ("", "data"):
            for suf in (".bin.zst", ".bin", ".raw"):
                p = os.path.join(self.root, sub, name + suf)
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(
            f"{name}.* не найден в {self.root} (искали ./ и data/, "
            f"суффиксы .bin.zst/.bin/.raw)")

    def _bits(self, k1, k2, kind):
        key = (k1, k2, kind)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self._find(f"layer_{k1}_{k2}_{kind}")
        with open(path, "rb") as fh:
            data = fh.read()
        # Автодетект формата: сырой битсет / zstd(битсет) / контейнер /
        # zstd(контейнер). Ожидаемый размер битсета известен точно.
        expected = (count_states(STONES - k1 - k2) + 7) // 8
        if data[:4] == ZSTD_MAGIC:
            # результат — битсет (expected байт) либо контейнер (не крупнее
            # битсета + запас на несжимаемые блоки и таблицу офсетов)
            max_size = expected + expected // 128 + (1 << 20)
            data = _unzstd(data, max_size=max_size)
        if len(data) != expected:
            parsed = _parse_container(data, expected)
            if parsed is None:
                raise ValueError(
                    f"{path}: после распаковки {len(data)} байт вместо "
                    f"{expected}, и это не блочный контейнер решателя — "
                    f"формат файла не распознан")
            data = parsed
        if len(data) != expected:
            raise ValueError(f"{path}: битсет {len(data)} байт, ожидается {expected}")
        self._cache[key] = data
        if len(self._cache) > self.max_layers:
            self._cache.popitem(last=False)
        return data

    def value(self, pos):
        """0 = LOSS, 1 = DRAW, 2 = WIN для стороны на ходу.
        Терминальные правила — зеркало Solver.cpp/query.cpp."""
        k1, k2, pits = pos
        if k1 >= 26: return 2          # казан >= 26 решает игру немедленно
        if k2 >= 26: return 0
        if sum(pits[:5]) == 0: return 0  # пустая сторона на ходу = поражение
        idx = rank(pits)
        w = self._bits(k1, k2, "win")
        if (w[idx >> 3] >> (idx & 7)) & 1: return 2   # LSB-first, как в
        d = self._bits(k1, k2, "draw")                # Solver.cpp/Oracle.h
        if (d[idx >> 3] >> (idx & 7)) & 1: return 1
        return 0


def legal_moves(pos):
    """Свои лунки — индексы 0..4; ход возможен из непустой лунки."""
    return [i for i in range(5) if pos[2][i] > 0]


def is_terminal_loss(pos):
    """Сторона на ходу уже проиграла по правилам игры, БЕЗ tablebase:
    казан соперника >= 26 или своя сторона пуста (Solver.cpp / query.cpp)."""
    return pos[1] >= 26 or sum(pos[2][:5]) == 0


def apply_move(pos, move):
    """Ход стороны на ходу. Возвращает позицию С ТОЧКИ ЗРЕНИЯ СОПЕРНИКА
    (канонический вид tablebase). Построчно эквивалентен
    BestemsheCore.h::ExecuteMoveAndFlip / Solver.h::execute_move_and_flip."""
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
    root_v = tb.value((0, 0, [5] * 10))  # якорь: доказана победа 2-го игрока
    if root_v != 0:
        print(f"ЯКОРЬ ПРОВАЛЕН: value(старт) = {root_v}, ожидается 0 "
              f"(LOSS — форсированная победа второго игрока)")
        return False
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