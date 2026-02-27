# Google Colab Training Guide — Connect 4 AlphaZero

This guide covers running the full training pipeline on Google Colab's free T4 GPU.

---

## Prerequisites

1. The code is pushed to a **public GitHub repository** (required for `git clone` in Colab)
2. You have a Google account with Google Drive access
3. You have opened [colab.research.google.com](https://colab.research.google.com)

---

## Step 1 — Set Up Colab Runtime

1. Open a new Colab notebook
2. Go to **Runtime → Change runtime type**
3. Set **Hardware accelerator** to **T4 GPU**
4. Click **Save**

---

## Step 2 — Mount Google Drive (Critical for Persistence)

Colab sessions disconnect after ~12 hours (sooner if idle). Without Drive, all checkpoints are lost on disconnect. Run this first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the authorization prompt. Your Drive will be mounted at `/content/drive/MyDrive/`.

---

## Step 3 — Clone Repo and Install Dependencies

```python
!git clone https://github.com/YOUR_USERNAME/connect4-alphazero.git
%cd connect4-alphazero
!pip install -e . -q
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Step 4 — Symlink Checkpoints to Google Drive

This is the key step that makes training survive session disconnects.
Use `os.symlink` with explicit absolute paths (avoids shell CWD issues):

```python
import os

DRIVE_CKPT = '/content/drive/MyDrive/connect4-checkpoints'
REPO_CKPT  = '/content/connect4-alphazero/checkpoints'

# Create checkpoint folder on Drive
os.makedirs(DRIVE_CKPT, exist_ok=True)

# Symlink: repo's checkpoints/ → Drive folder
# (only if the symlink doesn't already exist from a previous session)
if not os.path.exists(REPO_CKPT):
    os.symlink(DRIVE_CKPT, REPO_CKPT)

# Verify
print(os.path.islink(REPO_CKPT), '->', os.readlink(REPO_CKPT))
!ls /content/drive/MyDrive/connect4-checkpoints
```

From now on, every checkpoint saved by the training script goes straight to your Google Drive.

---

## Step 5 — Start Training

```python
%cd /content/connect4-alphazero
!python scripts/train.py --config configs/cloud.yaml \
  2>&1 | tee /content/drive/MyDrive/connect4-training.log
```

The `tee` command writes output both to the notebook cell AND to a log file on Drive. If the session disconnects, the log file preserves everything logged so far.

**Expected output per iteration:**
```
=== Iteration 1 / 20 ===
Self-play: generating 2000 games...
Replay buffer size: 4120
Training for 10 epochs...
Losses — policy: 1.923  value: 0.498  total: 2.421
Arena: 64 games (candidate vs. best)...
Arena result — wins: 0  losses: 0  draws: 0  win_rate: N/A (first iteration, auto-accepted)
Saved checkpoint: checkpoints/checkpoint_iter_000.pt
```

---

## Step 6 — Resume After Disconnect

When the session disconnects, re-run cells 1 (mount Drive) through 4 (symlink), then:

```python
import os

DRIVE_CKPT = '/content/drive/MyDrive/connect4-checkpoints'

# Find the latest checkpoint on Drive
checkpoints = sorted([f for f in os.listdir(DRIVE_CKPT) if f.startswith('checkpoint_iter_')])
latest = os.path.join(DRIVE_CKPT, checkpoints[-1])
print(f"Resuming from: {latest}")

!python /content/connect4-alphazero/scripts/train.py \
  --config /content/connect4-alphazero/configs/cloud.yaml \
  --resume {latest} 2>&1 | tee -a /content/drive/MyDrive/connect4-training.log
```

The `-a` flag appends to the existing log file rather than overwriting it.

---

## Step 7 — Monitor Training Health

Training is working correctly when:

| Signal | Iteration 1-3 | Iteration 5+ | Problem |
|---|---|---|---|
| `policy_loss` | ~1.8–1.9 | Decreasing | Stalls → learning rate too high |
| `value_loss` | ~0.4–0.6 | Decreasing | Stays ~0.33 → value head collapse |
| Arena `win_rate` | N/A (first iter auto-accepts) | Some iterations ≥ 0.55 | Never ≥ 0.55 → model not improving |

**Value head collapse** (value_loss stays near 0.33, model predicts ~0 for all positions): Restart with `learning_rate: 5.0e-4` in the config.

**First 3–5 iterations:** The agent barely beats random. This is normal — it takes several iterations to build a knowledge base. Meaningful improvement typically shows around iteration 8–12.

---

## Step 8 — Check Progress Anytime

Open a separate Colab cell (won't interrupt training) and run:

```python
# Show last 50 lines of training log
!tail -50 /content/drive/MyDrive/connect4-training.log

# List all checkpoints
!ls -lh /content/drive/MyDrive/connect4-checkpoints/

# Quick check: did the model improve this iteration?
!grep "Accepted\|Rejected" /content/drive/MyDrive/connect4-training.log | tail -10
```

---

## Step 9 — Download the Trained Model

**Option A — Download directly from Drive:**
- Open [drive.google.com](https://drive.google.com)
- Navigate to `connect4-checkpoints/`
- Right-click `best_model.pt` → Download

**Option B — Download via Colab:**
```python
from google.colab import files
files.download('/content/drive/MyDrive/connect4-checkpoints/best_model.pt')
```

**Option C — Copy back to your local machine using `gdown`** (if you know the Drive file ID):
```bash
# On your MacBook
pip install gdown
gdown https://drive.google.com/uc?id=FILE_ID -O checkpoints/best_model.pt
```

---

## Step 10 — Post-Training Evaluation (Local MacBook)

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

## Config Comparison

| Config | Model | Sims | Games/iter | Time/iter (T4) | Use |
|---|---|---|---|---|---|
| `tiny.yaml` | 2b/32f | 50 | 100 | ~5 min | Local unit tests |
| `small.yaml` | 3b/64f | 200 | 1000 | ~1-2h | Quick cloud validation |
| `cloud.yaml` | 5b/128f | 300 | 2000 | ~2-4h | **Recommended for Colab** |
| `full.yaml` | 5b/128f | 600 | 5000 | ~10-15h | Vast.ai only |

---

## Vast.ai Alternative (Paid, ~$10-20)

For a faster, uninterrupted full run on Vast.ai:

```bash
# 1. Rent an RTX 3090 instance (~$0.20/hr) with PyTorch template

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
tmux new -s train
python scripts/train.py --config configs/full.yaml 2>&1 | tee logs/training.log
# Ctrl+B then D to detach; tmux attach -t train to reattach

# 4. Monitor from local machine
ssh -p <port> root@<ip> "tail -f /workspace/connect4/logs/training.log"

# 5. Download results
rsync -av root@<ip>:<port-path>/workspace/connect4/checkpoints/ \
  /Users/eugenep/git/connect4-alphazero/checkpoints/
```

Vast.ai RTX 3090 with `full.yaml`: ~2-4h per iteration, ~50-80h total for 25 iterations.
