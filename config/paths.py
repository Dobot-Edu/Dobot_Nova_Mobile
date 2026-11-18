from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR = DATASETS_DIR / "raw"
LEROBOT_DATASET_DIR = DATASETS_DIR / "lerobot" / "nova_pi05_mobile_position_v3"
EXCLUDED_DATASET_DIR = DATASETS_DIR / "excluded"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# 数据流程使用的简短别名。
RAW_ROOT = RAW_DATASET_DIR
LEROBOT_ROOT = LEROBOT_DATASET_DIR
