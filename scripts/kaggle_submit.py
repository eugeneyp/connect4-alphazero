"""Package kaggle_agent.py + ONNX model into a Kaggle submission directory.

Reads src/export/kaggle_agent.py, replaces the _MODEL_PATH and _NUM_MCTS_SIMS
sentinel lines, writes submission/submission.py, and copies the ONNX model.

Usage:
    python scripts/kaggle_submit.py --model model.onnx --output submission/
    python scripts/kaggle_submit.py --model model.onnx --output submission/ \\
        --dataset-path /kaggle/input/connect4-alphazero-model/model.onnx \\
        --num-sims 200
"""

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_AGENT_SOURCE: Path = _PROJECT_ROOT / "src" / "export" / "kaggle_agent.py"

# Sentinel strings that appear exactly once in kaggle_agent.py
_MODEL_PATH_SENTINEL: str = '_MODEL_PATH: str = "model.onnx"'
_NUM_SIMS_SENTINEL: str = "_NUM_MCTS_SIMS: int = 200"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package kaggle_agent.py + ONNX model into a submission directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the exported .onnx model file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the submission package.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/kaggle/input/connect4-alphazero-model/model.onnx",
        help=(
            "Path to the model on Kaggle's filesystem "
            "(default: /kaggle/input/connect4-alphazero-model/model.onnx)."
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=200,
        help="Number of MCTS simulations per move (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    """Build the submission package."""
    args = _parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not _AGENT_SOURCE.exists():
        print(f"Error: agent source not found: {_AGENT_SOURCE}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the agent source template
    source = _AGENT_SOURCE.read_text(encoding="utf-8")

    # Validate sentinels exist before replacing (fail fast on template drift)
    if _MODEL_PATH_SENTINEL not in source:
        print(
            f"Error: sentinel not found in {_AGENT_SOURCE}:\n  {_MODEL_PATH_SENTINEL}",
            file=sys.stderr,
        )
        sys.exit(1)
    if _NUM_SIMS_SENTINEL not in source:
        print(
            f"Error: sentinel not found in {_AGENT_SOURCE}:\n  {_NUM_SIMS_SENTINEL}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Replace sentinel lines (each appears exactly once)
    source = source.replace(
        _MODEL_PATH_SENTINEL,
        f'_MODEL_PATH: str = "{args.dataset_path}"',
    )
    source = source.replace(
        _NUM_SIMS_SENTINEL,
        f"_NUM_MCTS_SIMS: int = {args.num_sims}",
    )

    # Write submission agent
    submission_py = output_dir / "submission.py"
    submission_py.write_text(source, encoding="utf-8")
    print(f"Wrote {submission_py}")

    # Copy the ONNX model
    dest_model = output_dir / model_path.name
    shutil.copy2(model_path, dest_model)
    print(f"Copied model to {dest_model}")

    print(f"\nSubmission package ready in: {output_dir}/")
    print(f"  submission.py  (_MODEL_PATH={args.dataset_path!r})")
    print(f"  {model_path.name}")
    print(
        "\nNext steps:"
        "\n  1. Upload model.onnx as a Kaggle Dataset."
        "\n  2. Submit submission.py as a Kaggle notebook agent."
    )


if __name__ == "__main__":
    main()
