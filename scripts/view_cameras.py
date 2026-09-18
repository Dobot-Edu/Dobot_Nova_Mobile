"""启动只通过 SSH 隧道访问的 top + 腕部相机查看器。"""

from Dobot_Init.dual_camera_viewer import load_dual_camera_config, run


if __name__ == "__main__":
    raise SystemExit(run(load_dual_camera_config()))
