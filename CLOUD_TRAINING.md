# Cloud Training Guide — Connect 4 AlphaZero

Training on cloud GPU. Choose your platform:

| Platform | GPU | Free hours | Disconnects? | Best for |
|---|---|---|---|---|
| **Kaggle** | P100 | 30h/week | No (Save & Run All) | Recommended |
| **Colab** | T4 | ~12h/session | Yes | Quick tests |
| **Vast.ai** | RTX 3090+ | Paid (~$0.20/hr) | No (tmux) | Full production run |

---

## Option A — Kaggle (Recommended)

Kaggle's **Save & Run All** runs your notebook to completion in a background worker — even if you close the browser. Output streams live so you can check in anytime.

### Setup

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. **Settings** (right panel) → Accelerator → **GPU P100** (not T4 x2 — our code is single-GPU only; the second T4 would be wasted)
3. **Settings** → Persistence → **Files only** (persists `/kaggle/working/` between sessions; variables are lost on completion anyway)
4. **Settings** → Internet → **On**

### Cell 1 — Clone and Install

```python
!git clone https://github.com/YOUR_USERNAME/connect4-alphazero.git /kaggle/working/connect4-alphazero
!pip install -e /kaggle/working/connect4-alphazero -q
```

### Cell 2 — Train

```python
%cd /kaggle/working/connect4-alphazero
!python scripts/train.py --config configs/cloud.yaml \
  2>&1 | tee /kaggle/working/training.log
```

Checkpoints are saved to `/kaggle/working/checkpoints/` automatically.

### Cell 3 — (Optional) Resume from a previous run

Upload your previous `checkpoint_iter_NNN.pt` as a Kaggle Dataset, then:

```python
import os, glob
checkpoints = sorted(glob.glob('/kaggle/input/YOUR-DATASET/checkpoint_iter_*.pt'))
latest = checkpoints[-1]
print(f"Resuming from: {latest}")

%cd /kaggle/working/connect4-alphazero
!python scripts/train.py --config configs/cloud.yaml --resume {latest} \
  2>&1 | tee /kaggle/working/training.log
```

### Running

Click **Save Version** → **Save & Run All** → **Save**. The notebook queues as a background job. You can close the browser and return later — output streams live when you revisit.

### Downloading Results

After the run completes, go to the notebook's **Output** tab:
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
!git clone https://github.com/YOUR_USERNAME/connect4-alphazero.git /content/connect4-alphazero
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

## Option C — Vast.ai (Paid, Full Production Run)

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

| Config | Model | Sims | Games/iter | Time/iter | Use |
|---|---|---|---|---|---|
| `tiny.yaml` | 2b/32f | 50 | 100 | ~5 min | Local unit tests |
| `small.yaml` | 3b/64f | 200 | 1000 | ~1-2h | Quick validation |
| `cloud.yaml` | 5b/128f | 100 | 200 | ~20 min | **Free cloud (Kaggle/Colab)** |
| `full.yaml` | 5b/128f | 600 | 5000 | ~2-4h | Vast.ai production run |
