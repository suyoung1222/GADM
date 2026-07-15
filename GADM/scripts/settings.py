import os
from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
# Override these locations with environment variables if your datasets,
# checkpoints, or results live outside the repository.
DATA_PATH = Path(os.environ.get("GADM_DATA_PATH", root / "data"))  # datasets and pretrained weights
TRAINING_PATH = Path(os.environ.get("GADM_TRAINING_PATH", root / "outputs/training"))  # training checkpoints
EVAL_PATH = Path(os.environ.get("GADM_EVAL_PATH", root / "outputs/results"))  # evaluation results
