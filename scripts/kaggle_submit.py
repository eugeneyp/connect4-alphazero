"""Package kaggle_agent.py + ONNX model into a Kaggle submission.

Three modes:

  Base64 mode (--base64, RECOMMENDED for Connect-X):
    Writes submission/main.py with the ONNX model embedded as a base64
    string. Upload main.py directly — no dataset, no zip, no notebook.
    On first call, the model is decoded to a temp file and loaded.

  Zip mode (--zip, DOES NOT WORK for Connect-X):
    Kaggle does not extract zip archives for Connect-X submissions.
    It places the raw zip bytes at /kaggle_simulations/agent/main.py and
    tries to compile them as Python, causing a SyntaxError.

  Directory mode (default):
    Writes submission/submission.py + model.onnx for notebook-based
    submissions that attach the model as a separate Kaggle Dataset.

Usage:
    # Base64 — single file upload (recommended for Connect-X)
    python scripts/kaggle_submit.py --model model.onnx --output submission/ --base64

    # Directory — for notebook submissions with a separate dataset
    python scripts/kaggle_submit.py --model model.onnx --output submission/ \\
        --dataset-path /kaggle/input/connect4-alphazero-model/model.onnx \\
        --num-sims 200
"""

import argparse
import base64
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

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--base64",
        action="store_true",
        default=False,
        help=(
            "Embed the ONNX model as base64 in a single main.py. "
            "Upload main.py directly to Kaggle Connect-X. (RECOMMENDED)"
        ),
    )
    mode.add_argument(
        "--zip",
        action="store_true",
        default=False,
        help=(
            "Create submission.zip with main.py + model.onnx. "
            "NOTE: does not work for Connect-X — Kaggle does not extract zips."
        ),
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/kaggle/input/connect4-alphazero-model/model.onnx",
        help=(
            "Model path on Kaggle's filesystem (directory mode only). "
            "Default: /kaggle/input/connect4-alphazero-model/model.onnx"
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=200,
        help="Number of MCTS simulations per move (default: 200).",
    )
    return parser.parse_args()


def _read_and_validate_source() -> str:
    """Read kaggle_agent.py and validate both sentinels are present."""
    source = _AGENT_SOURCE.read_text(encoding="utf-8")
    for sentinel in (_MODEL_PATH_SENTINEL, _NUM_SIMS_SENTINEL):
        if sentinel not in source:
            print(
                f"Error: sentinel not found in {_AGENT_SOURCE}:\n  {sentinel}",
                file=sys.stderr,
            )
            sys.exit(1)
    return source


def _apply_num_sims(source: str, num_sims: int) -> str:
    return source.replace(_NUM_SIMS_SENTINEL, f"_NUM_MCTS_SIMS: int = {num_sims}")


def _build_base64_submission(output_dir: Path, model_path: Path, num_sims: int) -> Path:
    """Create a single main.py with the ONNX model embedded as base64.

    The model bytes are decoded to a temp file on first agent call, so
    _MODEL_PATH resolves at runtime without any external file dependency.
    Upload main.py directly to Kaggle Connect-X submissions.
    """
    model_b64 = base64.b64encode(model_path.read_bytes()).decode("ascii")

    source = _read_and_validate_source()
    source = _apply_num_sims(source, num_sims)

    # Replace the _MODEL_PATH sentinel with a block that decodes the embedded
    # model to a temp file and sets _MODEL_PATH to that temp file's path.
    b64_block = (
        "# ONNX model embedded as base64 — no external file needed\n"
        "import base64 as _b64m, tempfile as _tmpm, atexit as _atm, os as _osm\n"
        f'_MODEL_DATA: str = "{model_b64}"\n'
        "_b64_tmp = _tmpm.NamedTemporaryFile(suffix='.onnx', delete=False)\n"
        "_b64_tmp.write(_b64m.b64decode(_MODEL_DATA))\n"
        "_b64_tmp.close()\n"
        "_MODEL_PATH: str = _b64_tmp.name\n"
        "_atm.register(lambda: _osm.unlink(_MODEL_PATH) if _osm.path.exists(_MODEL_PATH) else None)"
    )
    source = source.replace(_MODEL_PATH_SENTINEL, b64_block)

    output_path = output_dir / "main.py"
    output_path.write_text(source, encoding="utf-8")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    model_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path}  ({size_mb:.2f} MB)")
    print(f"\nSubmission ready: {output_path}")
    print(f"  Embedded model: {model_mb:.2f} MB → {size_mb:.2f} MB as base64")
    print(f"  MCTS sims:      {num_sims}")
    print("\nNext step: upload main.py directly to Kaggle → Submit Predictions.")
    return output_path


def _build_zip(output_dir: Path, model_path: Path, num_sims: int) -> Path:
    """Create submission.zip with main.py + model.onnx.

    WARNING: Kaggle Connect-X does not extract zip archives. This mode
    is kept for reference but --base64 is the correct approach.
    """
    source = _read_and_validate_source()
    source = _apply_num_sims(source, num_sims)
    source = source.replace(_MODEL_PATH_SENTINEL, '_MODEL_PATH: str = "model.onnx"')

    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", source)
        zf.write(model_path, arcname="model.onnx")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path}  ({size_mb:.2f} MB)")
    print("\nWARNING: Kaggle Connect-X does not extract zip archives.")
    print("Use --base64 instead for a single self-contained main.py.")
    return zip_path


def _build_directory(
    output_dir: Path,
    model_path: Path,
    dataset_path: str,
    num_sims: int,
) -> None:
    """Create submission/ directory with submission.py + model.onnx."""
    source = _read_and_validate_source()
    source = _apply_num_sims(source, num_sims)
    source = source.replace(_MODEL_PATH_SENTINEL, f'_MODEL_PATH: str = "{dataset_path}"')

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
        "\n  2. Submit submission.py via Kaggle notebook."
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

    if args.base64:
        _build_base64_submission(output_dir, model_path, args.num_sims)
    elif args.zip:
        _build_zip(output_dir, model_path, args.num_sims)
    else:
        _build_directory(output_dir, model_path, args.dataset_path, args.num_sims)


if __name__ == "__main__":
    main()
