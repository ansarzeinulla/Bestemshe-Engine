"""eval.py — проверка «сверхчеловечности» обученной модели.

  python eval.py --tb /workspace/tablebase \
      --model /workspace/ckpt/model_60000.pt --n 1_000_000 --games 10000
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm
from bestemshe_core import (Tablebase, sample_position, legal_moves,
                            apply_move, is_terminal_loss)
from make_shards import optimal_move_mask
from train import BestemsheNet


def load_model(path):
    ck = torch.load(path, map_location="cuda")
    a = ck["args"]
    m = BestemsheNet(a["width"], a["blocks"]).cuda().eval()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def predict(model, pos):
    k1, k2, pits = pos
    v, pol = model(torch.tensor([pits], device="cuda"),
                   torch.tensor([[k1 // 2, k2 // 2]], device="cuda"))
    return int(v.argmax()), pol[0].float().cpu().numpy()


def engine_move(model, pos):
    """Боевой движок: выбор хода по value детей (1 полуход) — БЕЗ tablebase.
    Терминальный шорткат как в query.cpp: ход, опустошающий соперника или
    доводящий казан до >= 26, — немедленная победа по правилам, модель не
    вызывается (казан ребёнка может быть >= 26 — вне домена Embedding(13))."""
    best, best_v = None, -1
    for m in legal_moves(pos):
        child = apply_move(pos, m)
        if is_terminal_loss(child):         # соперник проиграл — ходим сразу
            return m
        child_v, _ = predict(model, child)
        if 2 - child_v > best_v:            # инверсия WDL после смены стороны
            best, best_v = m, 2 - child_v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--games", type=int, default=10000)
    a = ap.parse_args()
    tb, model = Tablebase(a.tb), load_model(a.model)
    rng = np.random.default_rng(1)

    # 1) точность WDL и доля оптимальных ходов на случайных позициях
    v_hit = m_hit = m_tot = 0
    pbar = tqdm(range(1, a.n + 1), desc="позиции", unit="поз")
    for i in pbar:
        pos = sample_position(rng)
        v_pred, pol = predict(model, pos)
        v_hit += (v_pred == tb.value(pos))
        mask = optimal_move_mask(tb, pos)
        if mask:
            m_tot += 1
            m_hit += bool((mask >> int(np.argmax(pol))) & 1)
        if i % 1000 == 0:
            pbar.set_postfix(wdl_acc=f"{v_hit / i:.5f}",
                             opt_rate=f"{m_hit / max(1, m_tot):.5f}")
    print(f"wdl_acc = {v_hit / a.n:.5f}          (цель >= 0.999)")
    print(f"optimal_move_rate = {m_hit / m_tot:.5f} (цель >= 0.995)")

    # 2) партии движка против оракула: инцидент = неоптимальный ход
    #    в позиции, где движку теоретически положена победа или ничья
    incidents = 0
    gbar = tqdm(range(a.games), desc="партии", unit="партия")
    for g in gbar:
        pos = (0, 0, [5] * 10)              # стартовая позиция: по 5 камней
        for ply in range(400):
            moves = legal_moves(pos)
            if not moves or pos[0] >= 26 or pos[1] >= 26:
                break
            mask = optimal_move_mask(tb, pos)
            if (ply + g) % 2 == 0:          # ход движка (цвета чередуются)
                m = engine_move(model, pos)
                if tb.value(pos) >= 1 and not (mask >> m) & 1:
                    incidents += 1
            else:                           # оракул всегда ходит оптимально
                m = int(mask).bit_length() - 1
            pos = apply_move(pos, m)
        gbar.set_postfix(incidents=incidents)
    print(f"incidents = {incidents}          (цель: строго 0)")


if __name__ == "__main__":
    main()