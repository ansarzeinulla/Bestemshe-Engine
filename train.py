"""train.py — обучение ResMLP-дистиллята tablebase Бестемше.

Запуск (значения по умолчанию = раздел 5.1 плана):
  python train.py --data /workspace/shards --ckpt /workspace/ckpt \
      --batch 32768 --lr 3e-4 --steps 60000 --width 1024 --blocks 8 \
      --hub-repo <ваш_ник>/bestemshe-engine
60 000 шагов x 32 768 = ~2 млрд просмотренных примеров (~3-4 эпохи по 500 млн).

ИЗМЕНЕНИЯ (continuous learning):
  --resume <ckpt.pt>    — грузит веса модели, состояние оптимизатора AdamW
                          и номер шага, чтобы дообучение продолжалось, а не
                          начиналось заново (чекпоинт теперь хранит "optimizer").
  --consume-shards      — по завершении одного полного прохода по данным
                          (эпохи) физически удаляет .bin файлы шардов из
                          --data, освобождая место под новую генерацию.
                          Удаление один раз (после первой законченной эпохи).
  --hub-repo/--hub-token — при каждом чекпоинте на HF Hub теперь льётся не
                          только latest.pt, но и training_stats.json
                          (step, loss, wdl_acc, lr) для мониторинга на сайте.
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        self.files = sorted(glob.glob(os.path.join(folder, "shard_*.bin")))
        assert self.files, f"нет шардов в {folder}"
        self.mm = [np.memmap(f, dtype=np.uint8).reshape(-1, 14) for f in self.files]
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


def _delete_shards(paths):
    """Физически убирает .bin шарды с диска (место освобождается под новую
    генерацию). Уже открытые np.memmap продолжают работать после unlink —
    так ведёт себя POSIX, пока страницы держатся в памяти воркеров."""
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            continue
    tqdm.write(f"consume-shards: удалено {len(paths)} файлов из {os.path.dirname(paths[0]) if paths else '?'}")


def _push_to_hub(repo, token, ckpt_path, stats_path):
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=ckpt_path, path_in_repo="latest.pt",
                repo_id=repo, token=token or None)
    upload_file(path_or_fileobj=stats_path, path_in_repo="training_stats.json",
                repo_id=repo, token=token or None)


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
    ap.add_argument("--hub-token", default="")
    ap.add_argument("--resume", default="", help="путь к чекпоинту для дообучения")
    ap.add_argument("--consume-shards", action="store_true",
                    help="удалить .bin шарды из --data после первой законченной эпохи")
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

    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cuda")
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        step = ck.get("step", 0)
        tqdm.write(f"resume: {a.resume} -> шаг {step}")

    shards_consumed = False
    t_ckpt = time.time()
    pbar = tqdm(total=a.steps, initial=step, desc="обучение", unit="шаг")
    acc = 0.0
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
            pbar.update(1)
            if step % 200 == 0:
                acc = (v.argmax(1) == wdl).float().mean().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 wdl_acc=f"{acc:.4f}")
            if time.time() - t_ckpt > a.ckpt_min * 60 or step == a.steps:
                path = os.path.join(a.ckpt, f"model_{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": opt.state_dict(), "args": vars(a)}, path)
                tqdm.write(f"checkpoint: {path}")
                if a.hub_repo:              # резервная копия + метрики на HF Hub
                    stats_path = os.path.join(a.ckpt, "training_stats.json")
                    with open(stats_path, "w") as f:
                        json.dump({
                            "step": step,
                            "loss": loss.item(),
                            "wdl_acc": acc,
                            "lr": g["lr"],
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }, f, indent=2)
                    _push_to_hub(a.hub_repo, a.hub_token, path, stats_path)
                t_ckpt = time.time()
            if step >= a.steps:
                break
        else:
            # for-loop дошёл до конца без break -> закончился полный проход по данным
            if a.consume_shards and not shards_consumed:
                _delete_shards(ds.files)
                shards_consumed = True
            continue
        if a.consume_shards and not shards_consumed:
            _delete_shards(ds.files)
            shards_consumed = True
        break
    pbar.close()

    model.eval(); hit = tot = 0          # финальная валидация
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pits, kaz, wdl, mask in tqdm(vl, desc="валидация", unit="батч"):
            v, _ = model(pits.cuda(), kaz.cuda())
            hit += (v.argmax(1).cpu() == wdl).sum().item(); tot += len(wdl)
    if tot:
        val_acc = hit / tot
        print(f"VAL wdl_acc = {val_acc:.5f}   (цель >= 0.999)")
    else:
        val_acc = None   # val-split пуст (слишком маленький --data при таком --val-frac)
        print("VAL: пропущено — val-split пуст (0 записей)")

    # ФИНАЛЬНЫЙ пуш на HF Hub — безусловно, а не только по таймеру ckpt-min,
    # чтобы по завершении обучения на Hub гарантированно лежала последняя модель.
    final_path = os.path.join(a.ckpt, f"model_{step}.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "args": vars(a)}, final_path)
    if a.hub_repo:
        stats_path = os.path.join(a.ckpt, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "step": step,
                "loss": loss.item(),
                "wdl_acc": acc,
                "val_wdl_acc": val_acc,
                "lr": a.lr,
                "final": True,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        _push_to_hub(a.hub_repo, a.hub_token, final_path, stats_path)
        print(f"финальный чекпоинт (step={step}) залит на HF Hub: {a.hub_repo}")


if __name__ == "__main__":
    main()
