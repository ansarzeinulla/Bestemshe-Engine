# Bestemshe-Engine — команды для сервера (continuous learning, прогрессивный масштаб)

Полный прогон continuous-learning пайплайна на GPU-инстансе (Vast.ai/RunPod/аналог).
Стратегия: генерируем данные и обучаем модель ступенями возрастающего масштаба
(1 → 10 → 100 → ... → 100 млн позиций), на каждой ступени дообучаем (`--resume`)
предыдущий чекпоинт и освобождаем диск (`--consume-shards`). Всё улетает на HF Hub —
скачивать вручную не нужно, если передан `--hub-repo`/`--hub-token`.

> **Токен HF**: не коммитьте реальный токен в git. Передавайте через переменную
> окружения `HF_TOKEN` и флаг `--hub-token "$HF_TOKEN"`.

> **Важно про `--steps`**: раз мы каждый раз резюмируем (`--resume`) с прошлого
> чекпоинта, `--steps` — это АБСОЛЮТНЫЙ целевой номер шага, а не "ещё N шагов".
> Поэтому числа шагов ниже накопительные, и они же — суффикс в имени чекпоинта
> `model_<step>.pt`, который нужно подставлять в следующий `--resume`.

## Этап 0. tmux (обязательно — обрыв SSH не должен убивать обучение)

```bash
tmux new -s train
# после обрыва связи:
tmux attach -t train
```

## Этап 1. Системные зависимости

```bash
apt-get update && apt-get install -y tmux htop zstd git
pip install -U zstandard numpy tqdm torch huggingface_hub
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Этап 2. Код и токен

```bash
git clone https://github.com/ansarzeinulla/Bestemshe-Engine
cd Bestemshe-Engine
export HF_TOKEN=<ваш_HF_токен>
huggingface-cli login --token "$HF_TOKEN"
export HUB_REPO=ansarzeinulla/bestemshe-engine
```

## Этап 3. Датасет (tablebase) — ≈10–30 минут

```bash
hf download ansarzeinulla/bestemshe-tablebase \
    --repo-type dataset --local-dir /tablebase
du -sh /tablebase   # ожидается ~8.3-9.0 GB
export TB=/tablebase/layers/compressed
```

## Этап 4. Прогрессивный цикл: шарды → обучение → оценка

Мониторинг генерации в соседнем tmux-окне (`Ctrl+b, c` / `Ctrl+b, n` для навигации):

```bash
watch -n 2 cat /shards/generation_status.json
```

---

### Ступень 1 — `n=1` (первый запуск, `--mode overwrite`, абсолютный `--steps 1`)

```bash
python make_shards.py --tb "$TB" --out /shards --n 1 --shard-size 1 --workers 1 --mode overwrite

python train.py --data /shards --ckpt /ckpt \
    --batch 1 --steps 1 --val-frac 0 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_1.pt --n 1 --games 1 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 2 — `n=10` (`--steps 6` = 1+5, `--resume model_1.pt`)

```bash
python make_shards.py --tb "$TB" --out /shards --n 10 --shard-size 10 --workers 1 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_1.pt --batch 4 --steps 6 --val-frac 0.2 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_6.pt --n 10 --games 5 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 3 — `n=100` (`--steps 26` = 6+20)

```bash
python make_shards.py --tb "$TB" --out /shards --n 100 --shard-size 100 --workers 4 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_6.pt --batch 16 --steps 26 --val-frac 0.1 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_26.pt --n 100 --games 20 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 4 — `n=1_000` (`--steps 76` = 26+50)

```bash
python make_shards.py --tb "$TB" --out /shards --n 1_000 --shard-size 1_000 --workers 8 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_26.pt --batch 128 --steps 76 --val-frac 0.05 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_76.pt --n 1_000 --games 100 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 5 — `n=10_000` (`--steps 176` = 76+100)

```bash
python make_shards.py --tb "$TB" --out /shards --n 10_000 --shard-size 10_000 --workers 16 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_76.pt --batch 1024 --steps 176 --val-frac 0.02 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_176.pt --n 10_000 --games 500 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 6 — `n=100_000` (`--steps 476` = 176+300)

```bash
python make_shards.py --tb "$TB" --out /shards --n 100_000 --shard-size 100_000 --workers 24 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_176.pt --batch 4096 --steps 476 --val-frac 0.01 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_476.pt --n 50_000 --games 1000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 7 — `n=300_000` (`--steps 976` = 476+500)

```bash
python make_shards.py --tb "$TB" --out /shards --n 300_000 --shard-size 300_000 --workers 24 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_476.pt --batch 8192 --steps 976 --val-frac 0.01 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_976.pt --n 100_000 --games 2000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 8 — `n=1_000_000` (`--steps 1976` = 976+1000)

```bash
python make_shards.py --tb "$TB" --out /shards --n 1_000_000 --shard-size 1_000_000 --workers 32 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_976.pt --batch 16384 --steps 1976 --val-frac 0.005 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_1976.pt --n 200_000 --games 3000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 9 — `n=3_000_000` (`--steps 3976` = 1976+2000, 2 шарда по 1.5M)

```bash
python make_shards.py --tb "$TB" --out /shards --n 3_000_000 --shard-size 1_500_000 --workers 32 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_1976.pt --batch 16384 --steps 3976 --val-frac 0.005 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_3976.pt --n 300_000 --games 5000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 10 — `n=10_000_000` (`--steps 8976` = 3976+5000, 5 шардов по 2M)

```bash
python make_shards.py --tb "$TB" --out /shards --n 10_000_000 --shard-size 2_000_000 --workers 32 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_3976.pt --batch 32768 --steps 8976 --val-frac 0.002 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_8976.pt --n 500_000 --games 8000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 11 — `n=30_000_000` (`--steps 18976` = 8976+10000, 6 шардов по 5M)

```bash
python make_shards.py --tb "$TB" --out /shards --n 30_000_000 --shard-size 5_000_000 --workers 32 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_8976.pt --batch 32768 --steps 18976 --val-frac 0.002 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_18976.pt --n 1_000_000 --games 10000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

### Ступень 12 — `n=100_000_000` (`--steps 78976` = 18976+60000, 5 шардов по 20M — финальный масштаб)

```bash
python make_shards.py --tb "$TB" --out /shards --n 100_000_000 --shard-size 20_000_000 --workers 32 --mode append

python train.py --data /shards --ckpt /ckpt \
    --resume /ckpt/model_18976.pt --batch 32768 --steps 78976 --val-frac 0.002 --consume-shards \
    --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"

python eval.py --tb "$TB" --model /ckpt/model_78976.pt --n 1_000_000 --games 10000 \
    --out-dir /eval --hub-repo "$HUB_REPO" --hub-token "$HF_TOKEN"
```

Критерии приёмки (печатаются в отчёте `eval.py` и пишутся в `/eval/eval_results_<step>.json`,
опционально пушатся на HF Hub): `wdl_acc >= 0.999`, `optimal_move_rate >= 0.995`, `incidents == 0`.
Если на 12-й ступени метрики ниже целевых — продолжайте: сгенерируйте ещё данных
(`--mode append`, тот же `/shards`) и запустите ещё один `train.py --resume` с бОльшим `--steps`.

> Ступени 1-2 (`n=1`, `n=10`) — это чисто smoke-test пайплайна (генерация → обучение →
> оценка проходят без ошибок и всё улетает на HF), а не осмысленное обучение — на таких
> объёмах метрики ничего не значат. Реальная точность появится начиная примерно со
> ступени 8-10.

## Этап 5. Завершение и контроль расходов (обязательно!) — ≈10 минут

```bash
# Ctrl+b, d — отсоединиться от tmux, процессы продолжат работать
```

1. **Vast.ai**: `Destroy instance` (не просто `Stop` — остановленный инстанс
   продолжает тарифицировать диск).
2. **RunPod**: `Terminate pod` и удалите Volume, если он больше не нужен
   (остановленный под тарифицирует диск по $0.20/ГБ/мес).
3. На следующий день проверьте страницу **Billing** на обеих платформах —
   списания должны прекратиться.
