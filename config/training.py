"""Nova π0.5 训练参数。TRAIN_ARGUMENTS 的键对应 lerobot-train 的 CLI 参数。"""

from config import dataset, paths
from config.paths import OUTPUTS_DIR

TRAIN_OUTPUT = OUTPUTS_DIR / "pi05_nova_mobile"
POLICY_REPO_ID = "local/pi05_nova_mobile"
PRETRAINED_MODEL = "lerobot/pi05_base"
TRAIN_STEPS = 30000
SAVE_FREQUENCY = 5000
BATCH_SIZE = 16
NUM_WORKERS = 8
POLICY_CHUNK_SIZE = 25
POLICY_ACTION_STEPS = 25

# 供生成命令使用。路径按训练机上的实际位置填写。
TRAIN_ARGUMENTS = {
    'dataset.repo_id': dataset.DATASET_REPO_ID,
    'dataset.root': paths.LEROBOT_ROOT,
    'dataset.video_backend': 'pyav',
    'policy.type': 'pi05',
    'policy.pretrained_path': PRETRAINED_MODEL,
    'policy.repo_id': POLICY_REPO_ID,
    'policy.device': 'cuda',
    'policy.dtype': 'bfloat16',
    'policy.gradient_checkpointing': True,
    'policy.compile_model': False,
    'policy.train_expert_only': False,
    'policy.freeze_vision_encoder': False,
    'policy.chunk_size': POLICY_CHUNK_SIZE,
    'policy.n_action_steps': POLICY_ACTION_STEPS,
    'policy.use_relative_actions': False,
    'policy.push_to_hub': False,
    'batch_size': BATCH_SIZE,
    'steps': TRAIN_STEPS,
    'policy.scheduler_decay_steps': TRAIN_STEPS,
    'save_checkpoint': True,
    'save_freq': SAVE_FREQUENCY,
    'log_freq': 20,
    'num_workers': NUM_WORKERS,
    'wandb.enable': False,
    'job_name': 'nova_pi05_mobile',
    'output_dir': TRAIN_OUTPUT,
}
