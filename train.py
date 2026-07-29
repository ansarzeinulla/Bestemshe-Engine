"""train.py — обучение ResMLP-дистиллята tablebase Бестемше.

Запуск (значения по умолчанию = раздел 5.1 плана):
  python train.py --data /workspace/shards --ckpt /workspace/ckpt \
      --batch 32768 --lr 3e-4 --steps 60000 --width 1024 --blocks 8 \
      --hub-repo <ваш_ник>/bestemshe-engine
60 000 шагов x 32 768 = ~2 млрд просмотренных примеров (~3-4 эпохи по 500 млн).
"""
import argparse, glob, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, w)
        self.fc2 = nn.Linear(w, w)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class BestemsheNet(nn.Module):
    """Рекомендуемая модель (раздел 3.2): embeddings -> residual-MLP ->
    две головы: value (loss/draw/win) и policy (5 логитов ходов)."""

    def __init__(self, width=1024, blocks=8, emb=32):
        super().__init__()
        self.pit_emb = nn.Embedding(51, emb)   # камней в лунке: 0..50
        self.kaz_emb = nn.Embedding(13, emb)   # казан // 2: 0..12
        self.inp = nn.Linear(12 * emb, width)
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.value_head = nn.Linear(width, 3)
        self.policy_head = nn.Linear(width, 5)

    def forward(self, pits, kaz):
        x = torch.cat([self.pit_emb(pits).flatten(1),
                       self.kaz_emb(kaz).flatten(1)], dim=1)
        h = self.body(F.gelu(self.inp(x)))
        return self.value_head(h), self.policy_head(h)


class ShardData(Dataset):
    """14-байтовые записи make_shards.py через np.memmap — без загрузки в RAM."""

    def __init__(self, folder):
        files = sorted(glob.glob(os.path.join(folder, "shard_*.bin")))
        assert files, f"нет шардов в {folder}"
        self.mm = [np.memmap(f, dtype=np.uint8).reshape(-1, 14) for f in files]
        self.cum = np.cumsum([m.shape[0] for m in self.mm])

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        s = int(np.searchsorted(self.cum, i, side="right"))
        r = np.array(self.mm[s][i - (self.cum[s - 1] if s else 0)])
        pits = torch.from_numpy(r[:10].astype(np.int64))
        kaz = torch.from_numpy(r[10:12].astype(np.int64))
        mask = torch.tensor([(r[13] >> b) & 1 for b in range(5)],
                            dtype=torch.float32)
        return pits, kaz, int(r[12]), mask


def lr_at(step, base, warmup, total):
    """Warmup + косинусное затухание."""
    if step < warmup:
        return base * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="/workspace/ckpt")
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.002)
    ap.add_argument("--ckpt-min", type=int, default=30)
    ap.add_argument("--hub-repo", default="")
    a = ap.parse_args()
    os.makedirs(a.ckpt, exist_ok=True)

    ds = ShardData(a.data)
    n_val = int(len(ds) * a.val_frac)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=6)
    vl = DataLoader(va, batch_size=a.batch, num_workers=2)

    model = torch.compile(BestemsheNet(a.width, a.blocks).cuda())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    step, t_ckpt = 0, time.time()
    while step < a.steps:
        for pits, kaz, wdl, mask in dl:
            pits = pits.cuda(non_blocking=True)
            kaz = kaz.cuda(non_blocking=True)
            wdl = wdl.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.lr, a.warmup, a.steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v, pol = model(pits, kaz)
                loss = (F.cross_entropy(v, wdl) +
                        F.binary_cross_entropy_with_logits(pol, mask))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                acc = (v.argmax(1) == wdl).float().mean().item()
                print(f"step {step}  loss {loss.item():.4f}  wdl_acc {acc:.4f}")
            if time.time() - t_ckpt > a.ckpt_min * 60 or step == a.steps:
                path = os.path.join(a.ckpt, f"model_{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(a)}, path)
                if a.hub_repo:              # резервная копия на HF Hub
                    from huggingface_hub import upload_file
                    upload_file(path_or_fileobj=path, path_in_repo="latest.pt",
                                repo_id=a.hub_repo)
                t_ckpt = time.time()
            if step >= a.steps:
                break

    model.eval(); hit = tot = 0          # финальная валидация
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pits, kaz, wdl, mask in vl:
            v, _ = model(pits.cuda(), kaz.cuda())
            hit += (v.argmax(1).cpu() == wdl).sum().item(); tot += len(wdl)
    print(f"VAL wdl_acc = {hit / tot:.5f}   (цель >= 0.999)")


if __name__ == "__main__":
    main()