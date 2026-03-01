# Cloud Training Guide — Connect 4 AlphaZero

Training on cloud GPU. Choose your platform:

| Platform | GPU | Cost | Session limit | Best for |
|---|---|---|---|---|
| **Kaggle** | P100 | Free (30h/week) | ~9h/run | Recommended for free runs |
| **GCP** | L4 / T4 / V100 | ~$0.54–$3/hr | No limit | Best with $400 trial credit |
| **Colab** | T4 | Free (~12h/session) | ~12h | Quick tests |
| **Vast.ai** | RTX 3090+ | ~$0.20/hr | No limit | Cheapest paid option |

---

## Option A — Kaggle (Recommended)

Kaggle's **Save & Run All** runs your notebook to completion in a background worker — even if you close the browser. Output streams live so you can check in anytime.

**Session limit:** Committed runs have a ~9-hour hard cutoff. With `medium.yaml` (~100 min/iter) you'll get ~5-6 iterations per session. The checkpoints saved before cutoff are preserved in the Output tab — resume in a second session to finish the remaining iterations.

### Setup

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. **Settings** (right panel) → Accelerator → **GPU P100** (not T4 x2 — our code is single-GPU only; the second T4 would be wasted)
3. **Settings** → Persistence → **Files only** (persists `/kaggle/working/` between sessions; variables are lost on completion anyway)
4. **Settings** → Internet → **On**

### Cell 1 — Clone and Install

```python
!git clone https://github.com/eugeneyp/connect4-alphazero.git /kaggle/working/connect4-alphazero
!pip install /kaggle/working/connect4-alphazero -q
```

### Cell 2 — Train

```python
%cd /kaggle/working/connect4-alphazero
!python3 scripts/train.py --config configs/cloud.yaml \
  2>&1 | tee /kaggle/working/training.log
```

Checkpoints are saved to `/kaggle/working/checkpoints/` automatically.

### Running

Click **Save Version** → **Save & Run All** → **Save**. The notebook queues as a background job. You can close the browser and return later — output streams live when you revisit.

If you get a `ConcurrencyViolation` error, refresh the page and try again (stale sequence number from a previously stopped run).

### Resuming After Session Cutoff (~9h limit)

When the session times out, Kaggle saves everything in `/kaggle/working/` as the run's output.

**Step 1 — Publish the output as a dataset:**
- Go to the notebook → **Output** tab → **New Dataset**
- Give it a name (e.g. `connect4-checkpoints`) and publish it

**Step 2 — Add it as input to your next run:**
- Open the notebook → **Add data** → search for your dataset → add it
- It will appear at `/kaggle/input/connect4-checkpoints/`

**Step 3 — Find the exact dataset path (run once):**

```python
import os
for entry in os.listdir('/kaggle/input'):
    print(entry)
```

The path format is `/kaggle/input/datasets/<username>/<dataset-slug>/connect4-alphazero/checkpoints/`.

**Step 4 — Replace Cell 2 with the resume cell:**

```python
import glob

checkpoints = sorted(glob.glob('/kaggle/input/datasets/YOUR_USERNAME/connect4-checkpoints/connect4-alphazero/checkpoints/checkpoint_iter_*.pt'))
latest = checkpoints[-1]
print(f"Resuming from: {latest}")

%cd /kaggle/working/connect4-alphazero
!python3 scripts/train.py --config configs/medium.yaml --resume {latest} \
  2>&1 | tee /kaggle/working/training.log
```

### Running medium_v2 (Batched, Recommended)

Use `medium_v2.yaml` for all new Kaggle runs. It enables batched GPU inference (`mcts_batch_size: 32`), giving ~25 min/iter instead of ~100 min — fitting ~18 iterations per 9h session.

**Fresh start (Cell 2):**

```python
%cd /kaggle/working/connect4-alphazero
!python3 scripts/train.py --config configs/medium_v2.yaml \
  2>&1 | tee /kaggle/working/training.log
```

**Resuming from a previous medium run (e.g. checkpoint_iter_010.pt):**

Follow Steps 1–3 from "Resuming After Session Cutoff" above, then:

```python
import glob

checkpoints = sorted(glob.glob('/kaggle/input/datasets/YOUR_USERNAME/connect4-checkpoints/connect4-alphazero/checkpoints/checkpoint_iter_*.pt'))
latest = checkpoints[-1]
print(f"Resuming from: {latest}")

%cd /kaggle/working/connect4-alphazero
!python3 scripts/train.py --config configs/medium_v2.yaml --resume {latest} \
  2>&1 | tee /kaggle/working/training.log
```

The config has `num_iterations: 22` — resuming from iteration 10 (`checkpoint_iter_010.pt`) will run 11 more iterations (11 through 21).

### Downloading Results

After the final run completes, go to the notebook's **Output** tab:
- `checkpoints/best_model.pt` — download this
- `training.log` — download for analysis

---

## Option B — Google Colab

Colab sessions disconnect after ~12 hours. Use Google Drive to persist checkpoints across disconnects.

### Full Restart Cell (run this after every session reconnect)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone and install
!git clone https://github.com/eugeneyp/connect4-alphazero.git /content/connect4-alphazero
!pip install /content/connect4-alphazero -q

# Symlink checkpoints → Drive (survives disconnects)
import os
DRIVE_CKPT = '/content/drive/MyDrive/connect4-checkpoints'
REPO_CKPT  = '/content/connect4-alphazero/checkpoints'
os.makedirs(DRIVE_CKPT, exist_ok=True)
if not os.path.exists(REPO_CKPT):
    os.symlink(DRIVE_CKPT, REPO_CKPT)
print(os.path.islink(REPO_CKPT), '->', os.readlink(REPO_CKPT))
```

### Start Training

```python
%cd /content/connect4-alphazero
!python3 scripts/train.py --config configs/cloud.yaml \
  2>&1 | tee /content/drive/MyDrive/connect4-training.log
```

### Resume After Disconnect

Re-run the Full Restart Cell above, then:

```python
import os, glob
DRIVE_CKPT = '/content/drive/MyDrive/connect4-checkpoints'
checkpoints = sorted(glob.glob(f'{DRIVE_CKPT}/checkpoint_iter_*.pt'))
latest = checkpoints[-1]
print(f"Resuming from: {latest}")

%cd /content/connect4-alphazero
!python3 scripts/train.py --config configs/cloud.yaml --resume {latest} \
  2>&1 | tee -a /content/drive/MyDrive/connect4-training.log
```

### Selecting GPU

Go to **Runtime → Change runtime type** → set Hardware accelerator to **T4 GPU**.

### Downloading Results

```python
from google.colab import files
files.download('/content/drive/MyDrive/connect4-checkpoints/best_model.pt')
```

Or download directly from [drive.google.com](https://drive.google.com) → `connect4-checkpoints/best_model.pt`.

---

## Option C — GCP (Best Use of $400 Trial Credit)

GCP Compute Engine with a **Deep Learning VM** image (PyTorch + CUDA pre-installed, no setup needed). No session time limit — runs until you stop it or the VM is deleted.

### Recommended Instance

| Use case | Machine type | GPU | mcts_batch_size | Cost/hr | Total cost |
|---|---|---|---|---|---|
| `batched_test.yaml` (2 iters, validation) | `g2-standard-4` | 1× L4 | 32 | ~$0.90 | < $1 |
| `full.yaml` (20 iters, production) | `g2-standard-4` | 1× L4 | 32 | ~$0.90 | ~$36 |

With $400 credit: an L4 instance runs for **400+ hours** — more than enough.

### One-Time Setup

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable Compute Engine API (once per project)
gcloud services enable compute.googleapis.com
```

### Find the Current PyTorch Image

The Deep Learning VM image name changes with each PyTorch/CUDA release. Find the latest:

```bash
gcloud compute images list --project=deeplearning-platform-release --no-standard-images --filter="family~pytorch" --format="table(family,name,creationTimestamp)" --sort-by="~creationTimestamp" | head -5
```

Use the `family` value from the most recent result (e.g. `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`).

### Check L4 Zone Availability

```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="table(name,zone)"
```

Pick a zone from the output (ideally `us-central1-*`).

### Create the VM

L4 GPUs use `g2-standard` machine types — the GPU is built-in, no `--accelerator` flag needed.

```bash
gcloud compute instances create connect4-fullrun --zone=ZONE --machine-type=g2-standard-4 --image-family=IMAGE_FAMILY --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --restart-on-failure --boot-disk-size=100GB --metadata=install-nvidia-driver=True
```

Replace `ZONE` and `IMAGE_FAMILY` with values from the steps above.

### SSH In and Verify GPU

```bash
gcloud compute ssh connect4-fullrun --zone=ZONE
```

On the VM:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA L4
```

### Clone, Install, and Run

```bash
git clone https://github.com/eugeneyp/connect4-alphazero.git && cd connect4-alphazero && pip install . -q && mkdir -p logs
```

Run in tmux so training survives SSH disconnects:

```bash
tmux new -s fullrun
```

Inside tmux:

```bash
python3 scripts/train.py --config configs/full.yaml 2>&1 | tee logs/full_run.log
```

Detach: `Ctrl+B` then `D`. Reattach later: `tmux attach -t fullrun`.

### Monitor from Local Machine

```bash
gcloud compute ssh connect4-fullrun --zone=ZONE --command="tail -30 ~/connect4-alphazero/logs/full_run.log"
```

**Key log line to watch after each self-play phase:**
```
Self-play done: 2000 games in XX.X min (XX.X games/min)
```
Baseline (Kaggle P100, serial): ~8–9 games/min. Target (L4, batch_size=32): ~20 games/min.

### Resume After Disconnect

```bash
gcloud compute ssh connect4-fullrun --zone=ZONE
cd connect4-alphazero
tmux attach -t fullrun   # reattach if tmux session still alive

# Or restart from latest checkpoint:
LATEST=$(ls -t checkpoints/checkpoint_iter_*.pt | head -1)
python3 scripts/train.py --config configs/full.yaml --resume $LATEST 2>&1 | tee -a logs/full_run.log
```

### Download Results When Done

```bash
gcloud compute scp --recurse connect4-fullrun:~/connect4-alphazero/checkpoints/ /Users/eugenep/git/connect4-alphazero/checkpoints/ --zone=ZONE

gcloud compute scp connect4-fullrun:~/connect4-alphazero/logs/full_run.log /Users/eugenep/git/connect4-alphazero/logs/ --zone=ZONE
```

### Stop the VM When Done (avoids charges)

```bash
# Stop (keeps disk, ~$4/month)
gcloud compute instances stop connect4-fullrun --zone=ZONE

# Delete entirely when no longer needed
gcloud compute instances delete connect4-fullrun --zone=ZONE
```

---

## Option D — Vast.ai (Paid, Full Production Run)

```bash
# 1. Rent RTX 3090 instance (~$0.20/hr) with PyTorch template on vast.ai

# 2. Upload code from local machine
rsync -av \
  --exclude='checkpoints/' --exclude='logs/' --exclude='__pycache__/' \
  --exclude='.git/' --exclude='*.onnx' \
  /Users/eugenep/git/connect4-alphazero/ \
  root@<ip>:<port-path>/workspace/connect4/

# 3. SSH in and run with tmux (survives SSH disconnect)
ssh -p <port> root@<ip>
cd /workspace/connect4
pip install . -q
mkdir -p logs
tmux new -s train
python3 scripts/train.py --config configs/full.yaml 2>&1 | tee logs/training.log
# Ctrl+B then D to detach; tmux attach -t train to reattach

# 4. Monitor from local machine
ssh -p <port> root@<ip> "tail -f /workspace/connect4/logs/training.log"

# 5. Download results when done
rsync -av root@<ip>:<port-path>/workspace/connect4/checkpoints/ \
  /Users/eugenep/git/connect4-alphazero/checkpoints/
```

---

## Parallelizing Training

### Where the Time Goes

Self-play dominates (~85% of each iteration). Standard MCTS calls the NN once per simulation (batch size 1), leaving the GPU idle ~95% of the time.

| Phase | full.yaml (L4) | Parallelizable? |
|---|---|---|
| Self-play (MCTS) | ~100 min | ✅ Yes — batched MCTS |
| NN training | ~5 min | ✅ Yes — multi-GPU DataParallel |
| Arena | ~15 min | Partially |
| Benchmark | ~5 min | Partially |

### Option 1 — Batched MCTS Inference (Recommended ✅ Implemented)

Run M MCTS trees in lock-step so all leaf evaluations are batched into a single GPU forward pass per simulation step. GPU utilisation goes from ~5% to ~80%+.

**How to enable:** Add `mcts_batch_size` to any config's `training:` section:

```yaml
training:
  mcts_batch_size: 32   # number of games advancing in lock-step
```

The default is `1` (serial, unchanged behavior). All existing configs work as before.

**Mutually exclusive with `num_self_play_workers > 1`** — don't set both.

**Validate before a long run** using `configs/batched_test.yaml` (medium architecture, 2 iterations, batch_size=32):

```bash
python3 scripts/train.py --config configs/batched_test.yaml
```

Look for the `Self-play done: ... games/min` log line. Baseline (serial P100): ~8–9 games/min. Target (L4, batch_size=32, full.yaml): ~20 games/min.

### Option 2 — Parallel Self-Play Workers (CPU inference)

Run N games simultaneously using Python `multiprocessing`. Each worker uses a CPU copy of the model. Best for multi-core CPU-only setups or small models where CPU inference is fast.

**How to enable:**

```yaml
training:
  num_self_play_workers: 6   # for 8-vCPU instance, leave 2 for main + OS
```

**Not recommended alongside batched MCTS** — the two approaches are mutually exclusive. Use batched MCTS when a GPU is available; use parallel workers for CPU-only training.

### Option 3 — Multi-GPU DataParallel (Low Impact)

`torch.nn.DataParallel` splits each training batch across multiple GPUs. Since training takes only ~5 min per iteration, this gives minimal overall speedup. Not worth the complexity.

### Summary

| Optimization | Speedup vs serial | Status |
|---|---|---|
| Batched MCTS (GPU, batch_size=32) | ~6× on L4 | **Done ✅** |
| Parallel self-play workers (CPU) | 4–8× on multi-core | **Done ✅** |
| Multi-GPU DataParallel | ~1.1× | Not worth it |

---

## Monitoring Training Health

Applies to all platforms:

| Signal | Iteration 1-3 | Iteration 5+ | Problem |
|---|---|---|---|
| `policy_loss` | ~1.8–1.9 | Decreasing | Stalls → learning rate too high |
| `value_loss` | ~0.4–0.6 | Decreasing | Stays ~0.33 → value head collapse |
| Arena `win_rate` | N/A (first iter auto-accepts) | Some iterations ≥ 0.55 | Never ≥ 0.55 → not improving |
| Benchmark vs Random | ~50% (random model) | Increasing toward 95%+ | Flat → training not working |

**Arena win_rate** is calculated over decisive games only (wins / (wins + losses), draws excluded). A result of 10W/1L/19D = 0.91 win_rate → accepted.

**Value head collapse** (value_loss stuck near 0.33): restart with `learning_rate: 5.0e-4`.

**First 3–5 iterations:** The agent barely beats random — this is normal. Meaningful improvement shows around iteration 8–12.

---

## Post-Training Evaluation (Local MacBook)

After downloading `best_model.pt` to `checkpoints/`:

```bash
# Benchmark against all classical agents
python3 scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --num-games 100 \
  --depth 1 3 5 \
  --mcts-sims 400

# Export to ONNX for Kaggle submission
python3 scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pt \
  --output model.onnx

# Package and submit to Kaggle
python3 scripts/kaggle_submit.py \
  --model model.onnx \
  --output submission/
```

**Target win rates after 15-20 iterations:**
- vs Random: > 95%
- vs Minimax(depth=1): > 80%
- vs Minimax(depth=3): > 60%
- vs Minimax(depth=5): > 50%

---

## Config Reference

| Config | Model | Sims | Games/iter | batch_size | Time/iter | Use |
|---|---|---|---|---|---|---|
| `tiny.yaml` | 2b/32f | 50 | 100 | 1 | ~5 min | Local unit tests |
| `cloud.yaml` | 5b/128f | 100 | 200 | 1 | ~20 min | Pipeline validation |
| `medium.yaml` | 4b/64f | 200 | 800 | 1 | ~100 min (P100) | Kaggle serial run (legacy) |
| `medium_v2.yaml` | 4b/64f | 300 | 800 | 32 | ~25 min (P100) | **Kaggle batched run (recommended)** |
| `batched_test.yaml` | 4b/64f | 200 | 800 | 32 | ~20 min (L4) | Validate batched MCTS on GCP |
| `small.yaml` | 3b/64f | 200 | 1000 | 1 | ~1-2h | Alternative medium run |
| `full.yaml` | 5b/128f | 400 | 2000 | 32 | ~2h (L4) | **GCP production run** |
