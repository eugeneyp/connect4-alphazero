"""Package kaggle_agent.py + ONNX model into a Kaggle submission.

Two modes:
  Directory mode (default): writes submission/submission.py + model.onnx.
    Use when submitting via Kaggle notebook with a separately-uploaded dataset.

  Zip mode (--zip): writes submission/submission.zip containing main.py +
    model.onnx. Upload the zip directly to Kaggle — no separate dataset needed.
    Kaggle extracts the archive and runs main.py; model.onnx is in the same
    directory, so _MODEL_PATH = "model.onnx" (the default) works as-is.

Usage:
    # Zip archive — upload submission.zip directly (recommended)
    python scripts/kaggle_submit.py --model model.onnx --output submission/ --zip

    # Directory — for notebook-based submissions with a separate dataset
    python scripts/kaggle_submit.py --model model.onnx --output submission/ \\
        --dataset-path /kaggle/input/connect4-alphazero-model/model.onnx \\
        --num-sims 200
"""

import argparse
import shutil
import sys
import zipfile
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
        description="Package kaggle_agent.py + ONNX model into a Kaggle submission."
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
        "--zip",
        action="store_true",
        default=False,
        help=(
            "Create a zip archive (main.py + model.onnx) for direct Kaggle upload. "
            "_MODEL_PATH is kept as 'model.onnx' (relative, works inside the archive)."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/kaggle/input/connect4-alphazero-model/model.onnx",
        help=(
            "Path to the model on Kaggle's filesystem, used in directory mode "
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


def _build_source(num_sims: int, model_path_str: str) -> str:
    """Read kaggle_agent.py and replace sentinel lines.

    Args:
        num_sims: Number of MCTS simulations to bake in.
        model_path_str: The _MODEL_PATH value to write into the source.

    Returns:
        Modified source code as a string.
    """
    source = _AGENT_SOURCE.read_text(encoding="utf-8")

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

    source = source.replace(
        _MODEL_PATH_SENTINEL,
        f'_MODEL_PATH: str = "{model_path_str}"',
    )
    source = source.replace(
        _NUM_SIMS_SENTINEL,
        f"_NUM_MCTS_SIMS: int = {num_sims}",
    )
    return source


def _build_zip(output_dir: Path, model_path: Path, num_sims: int) -> Path:
    """Create a zip archive with main.py + model.onnx for direct Kaggle upload.

    Kaggle extracts the archive alongside main.py, so _MODEL_PATH = "model.onnx"
    (a plain filename, no directory prefix) works without a separate dataset.
    """
    source = _build_source(num_sims, model_path_str="model.onnx")

    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", source)
        zf.write(model_path, arcname="model.onnx")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path}  ({size_mb:.2f} MB)")
    print(f"\nSubmission zip ready: {zip_path}")
    print("  main.py        (agent, _MODEL_PATH='model.onnx')")
    print(f"  model.onnx     ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print("\nNext step: upload submission.zip directly to Kaggle → Submit Predictions.")
    return zip_path


def _build_directory(
    output_dir: Path,
    model_path: Path,
    dataset_path: str,
    num_sims: int,
) -> None:
    """Create submission/ directory with submission.py + model.onnx."""
    source = _build_source(num_sims, model_path_str=dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    submission_py = output_dir / "submission.py"
    submission_py.write_text(source, encoding="utf-8")
    print(f"Wrote {submission_py}")

    dest_model = output_dir / model_path.name
    shutil.copy2(model_path, dest_model)
    print(f"Copied model to {dest_model}")

    print(f"\nSubmission package ready in: {output_dir}/")
    print(f"  submission.py  (_MODEL_PATH={dataset_path!r})")
    print(f"  {model_path.name}")
    print(
        "\nNext steps:"
        "\n  1. Upload model.onnx as a Kaggle Dataset."
        "\n  2. Submit submission.py as a Kaggle notebook agent."
    )


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

    if args.zip:
        _build_zip(output_dir, model_path, args.num_sims)
    else:
        _build_directory(output_dir, model_path, args.dataset_path, args.num_sims)


if __name__ == "__main__":
    main()
