# Cloud Training Guide — Connect 4 AlphaZero

Training on cloud GPU. Choose your platform:

| Platform | GPU | Cost | Session limit | Best for |
|---|---|---|---|---|
| **Kaggle** | P100 | Free (30h/week) | ~9h/run | Recommended for free runs |
| **GCP** | T4 / V100 / A100 | ~$0.35–$3/hr | No limit | Best with $400 trial credit |
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
!git clone https://github.com/eugenep/connect4-alphazero.git /kaggle/working/connect4-alphazero
!pip install -e /kaggle/working/connect4-alphazero -q
```

### Cell 2 — Train

```python
%cd /kaggle/working/connect4-alphazero
!python scripts/train.py --config configs/cloud.yaml \
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
!python scripts/train.py --config configs/medium.yaml --resume {latest} \
  2>&1 | tee /kaggle/working/training.log
```

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
!git clone https://github.com/eugenep/connect4-alphazero.git /content/connect4-alphazero
!pip install -e /content/connect4-alphazero -q

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
!python scripts/train.py --config configs/cloud.yaml \
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
!python scripts/train.py --config configs/cloud.yaml --resume {latest} \
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

| Use case | Machine type | GPU | Cost/hr |
|---|---|---|---|
| `medium.yaml` (10 iters) | `n1-standard-8` | 1× T4 | ~$0.35 |
| `full.yaml` (25 iters) | `n1-standard-8` | 1× V100 | ~$1.10 |
| Fast full run | `n1-standard-16` | 1× A100 | ~$2.93 |

With $400 credit: a T4 instance runs for **1000+ hours**. A V100 runs for **360+ hours**. Either is more than enough.

### Create the VM

```bash
# Install gcloud CLI if needed: https://cloud.google.com/sdk/docs/install
# Then authenticate:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create a Deep Learning VM with T4 GPU (PyTorch pre-installed)
gcloud compute instances create connect4-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=50GB \
  --metadata="install-nvidia-driver=True"
```

For V100 instead, replace `nvidia-tesla-t4` with `nvidia-tesla-v100`.

### SSH and Run Training

```bash
# SSH in (gcloud handles key management automatically)
gcloud compute ssh connect4-training --zone=us-central1-a

# On the VM:
git clone https://github.com/eugenep/connect4-alphazero.git
cd connect4-alphazero
pip install -e . -q
mkdir -p logs

# Run in tmux (survives SSH disconnect)
tmux new -s train
python scripts/train.py --config configs/full.yaml 2>&1 | tee logs/training.log
# Ctrl+B then D to detach; tmux attach -t train to reattach
```

### Monitor from Local Machine

```bash
gcloud compute ssh connect4-training --zone=us-central1-a \
  --command="tail -f /home/$(whoami)/connect4-alphazero/logs/training.log"
```

### Download Results

```bash
gcloud compute scp --recurse \
  connect4-training:/home/$(whoami)/connect4-alphazero/checkpoints/ \
  /Users/eugenep/git/connect4-alphazero/checkpoints/ \
  --zone=us-central1-a
```

### Stop the VM When Done (avoids charges)

```bash
gcloud compute instances stop connect4-training --zone=us-central1-a
# Delete entirely when no longer needed:
gcloud compute instances delete connect4-training --zone=us-central1-a
```

---

## Option D — Vast.ai (Paid, Full Production Run)

For the full `full.yaml` run (600 sims × 5000 games × 25 iterations, ~$15-20 total).

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
pip install -e . -q
mkdir -p logs
tmux new -s train
python scripts/train.py --config configs/full.yaml 2>&1 | tee logs/training.log
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

Self-play dominates (~85% of each iteration). The GPU is underutilized because MCTS is a sequential Python loop — each simulation calls the NN once, serially. The GPU sits idle between calls.

| Phase | medium.yaml | Parallelizable? |
|---|---|---|
| Self-play (MCTS) | ~87 min | ✅ Yes — games are independent |
| NN training | ~3 min | ✅ Yes — multi-GPU DataParallel |
| Arena | ~10 min | Partially |
| Benchmark | ~8 min | Partially |

### Option 1 — Parallel Self-Play Workers (High Impact, Requires Code Change)

The biggest speedup: run N games simultaneously using Python `multiprocessing`. Each worker process plays one game independently. With 4–8 workers on a multi-core instance, self-play becomes 4–8× faster.

**Tradeoff:** Each worker needs its own model copy. On CPU this is fine (model is only 6MB). On GPU, workers share CUDA context which requires `torch.multiprocessing` with `spawn` start method.

**Practical approach for GCP:** Use an 8-core instance (`n1-standard-8`), run 4 CPU workers for self-play, and keep the GPU exclusively for the training step. Even with CPU inference, 4 parallel games typically beats 1 GPU game because Python overhead (not GPU compute) is the bottleneck at batch size 1.

This is listed in the project's performance optimization checklist (`CLAUDE.md` §14) and is the next planned optimization.

### Option 2 — Batched MCTS Inference (High Impact, Complex)

Instead of 1 NN call per simulation, collect multiple pending leaf nodes across parallel MCTS trees and batch them into a single GPU call. This is how the original AlphaZero achieves high GPU utilization.

Speedup: 5–10× GPU throughput. Complexity: significant refactor of `src/mcts/search.py`.

### Option 3 — Multi-GPU DataParallel (Low Impact for This Workload)

`torch.nn.DataParallel` splits each training batch across multiple GPUs. Since training is already fast (~3 min), this gives minimal overall speedup. Not worth the complexity unless the model grows much larger.

### Summary

| Optimization | Speedup | Effort | Status |
|---|---|---|---|
| Parallel self-play workers | 4–8× | Medium | Planned |
| Batched MCTS inference | 5–10× | High | Future |
| Multi-GPU DataParallel | ~1.1× | Low | Not worth it |

For now, the most practical path with $400 GCP credit is to run `full.yaml` on a single V100 (fast enough as-is) and implement parallel self-play workers as the next code improvement.

---

## Monitoring Training Health

Applies to all platforms:

| Signal | Iteration 1-3 | Iteration 5+ | Problem |
|---|---|---|---|
| `policy_loss` | ~1.8–1.9 | Decreasing | Stalls → learning rate too high |
| `value_loss` | ~0.4–0.6 | Decreasing | Stays ~0.33 → value head collapse |
| Arena `win_rate` | N/A (first iter auto-accepts) | Some iterations ≥ 0.55 | Never ≥ 0.55 → not improving |
| Benchmark vs Random | ~50% (random model) | Increasing toward 95%+ | Flat → training not working |

**Value head collapse** (value_loss stuck near 0.33): restart with `learning_rate: 5.0e-4`.

**First 3–5 iterations:** The agent barely beats random — this is normal. Meaningful improvement shows around iteration 8–12.

---

## Post-Training Evaluation (Local MacBook)

After downloading `best_model.pt` to `checkpoints/`:

```bash
# Benchmark against all classical agents
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --num-games 100 \
  --depth 1 3 5 \
  --mcts-sims 400

# Export to ONNX for Kaggle submission
python scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pt \
  --output model.onnx

# Package and submit to Kaggle
python scripts/kaggle_submit.py \
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

| Config | Model | Sims | Games/iter | Time/iter (P100) | Use |
|---|---|---|---|---|---|
| `tiny.yaml` | 2b/32f | 50 | 100 | ~5 min | Local unit tests |
| `cloud.yaml` | 5b/128f | 100 | 200 | ~20 min | Pipeline validation |
| `medium.yaml` | 4b/64f | 200 | 800 | ~100 min | **Kaggle/Colab training run** |
| `small.yaml` | 3b/64f | 200 | 1000 | ~1-2h | Alternative medium run |
| `full.yaml` | 5b/128f | 600 | 5000 | ~2-4h | Vast.ai production run |
