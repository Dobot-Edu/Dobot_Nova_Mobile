#!/usr/bin/env python3
"""把成功的原始示教转换为标准 LeRobot Dataset v3.0。"""

from __future__ import annotations

import argparse
import json
import queue
import time
import sys
import traceback
from pathlib import Path

import cv2

from config import dataset, paths
from Dobot_Train.lerobot_source import use_local_lerobot
from Dobot_Collection.schema import (
    ACTION_NAMES,
    STATE_NAMES,
    completed_episodes,
    episode_vectors,
    read_jsonl,
    validate_training_metadata,
)




def offline_dataset_class():
    """只在本脚本启用有界阻塞队列，避免实时采集编码器的满队列丢帧。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import StreamingVideoEncoder

    class OfflineVideoEncoder(StreamingVideoEncoder):
        def _put(self, key, value):
            while self._threads[key].is_alive():
                try:
                    self._frame_queues[key].put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue
            try:
                status, detail = self._result_queues[key].get_nowait()
            except queue.Empty:
                detail = "编码线程已停止"
            raise RuntimeError(f"{key}: {detail}")

        def feed_frame(self, video_key, image):
            if not self._episode_active:
                raise RuntimeError("视频episode尚未启动")
            # 转换器每次解码得到新数组，入队后不修改它，无需再次复制图像。
            self._put(video_key, image)

        def finish_episode(self):
            if not self._episode_active:
                raise RuntimeError("视频episode尚未启动")
            try:
                for key in self._frame_queues:
                    self._put(key, None)
                results = {}
                for key, thread in self._threads.items():
                    thread.join(timeout=120)
                    if thread.is_alive():
                        raise RuntimeError(f"{key}: 视频编码收尾超时")
                    try:
                        status, stats = self._result_queues[key].get(timeout=5)
                    except queue.Empty as error:
                        raise RuntimeError(f"{key}: 缺少编码结果") from error
                    if status != "ok":
                        raise RuntimeError(f"{key}: {stats}")
                    results[key] = (self._video_paths[key], stats)
                self._cleanup()
                self._episode_active = False
                return results
            except BaseException:
                self.cancel_episode()
                raise

    class OfflineDataset(LeRobotDataset):
        @staticmethod
        def _build_streaming_encoder(fps, rgb_encoder, depth_encoder,
                                     encoder_queue_maxsize, encoder_threads):
            return OfflineVideoEncoder(
                fps=fps, rgb_encoder=rgb_encoder, depth_encoder=depth_encoder,
                queue_maxsize=encoder_queue_maxsize, encoder_threads=encoder_threads,
            )

    return OfflineDataset


def convert_dataset(
    raw_root: Path = paths.RAW_ROOT,
    output_root: Path = paths.LEROBOT_ROOT,
    repo_id: str = dataset.DATASET_REPO_ID,
    *,
    use_videos: bool = dataset.USE_VIDEOS,
    video_batch_size: int = 1,
    image_writer_threads: int = 4,
    video_crf: float = 30,
    video_preset: str | None = "fast",
    streaming_encoding: bool = True,
    encoder_threads: int = 2,
    encoder_queue_maxsize: int = 16,
) -> tuple[int, int]:
    started = time.monotonic()
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if video_batch_size < 1 or image_writer_threads < 0 or not 0 <= video_crf <= 51:
        raise ValueError("编码批次须>=1，写图线程须>=0，CRF须在0..51之间")
    if encoder_threads < 1 or encoder_queue_maxsize < 1:
        raise ValueError("编码线程数和队列长度须>=1")
    streaming_encoding = bool(use_videos and streaming_encoding)
    # 本地LeRobot批量路径在写入模式读取meta.episodes（此时为None）。
    # 使用逐episode保存；save_episode仍默认并行编码两个相机。
    if use_videos and video_batch_size > 1:
        print("[编码] 当前LeRobot不支持此批量写入路径，改用逐条保存、双相机并行编码", flush=True)
        video_batch_size = 1
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在：{output_root}")
    episodes = completed_episodes(raw_root, successful_only=True)
    if not episodes:
        raise ValueError("没有带COMPLETE且success=true的原始episode")
    print(f"开始转换：{len(episodes)}条，输出：{output_root}", flush=True)
    use_local_lerobot()
    from lerobot.configs.video import RGBEncoderConfig
    LeRobotDataset = offline_dataset_class()
    print(f"[编码] {'流式，原始JPG直接解码入队' if streaming_encoding else '中间图片模式'}；"
          f"每路线程={encoder_threads}，队列={encoder_queue_maxsize}，preset={video_preset}", flush=True)

    first_row = next(read_jsonl(episodes[0] / "samples.jsonl"), None)
    if first_row is None:
        raise ValueError(f"{episodes[0].name} 没有样本")
    first_image = cv2.imread(str(episodes[0] / first_row["images"]["top"]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError("首帧Top图像无法读取")
    height, width = first_image.shape[:2]
    visual_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
        "observation.images.top": {
            "dtype": visual_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": visual_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=dataset.DATASET_FPS,
        features=features,
        root=output_root,
        robot_type="dobot_nova_mobile_lift_vr",
        use_videos=use_videos,
        image_writer_threads=0 if streaming_encoding else image_writer_threads,
        streaming_encoding=streaming_encoding,
        encoder_threads=encoder_threads,
        encoder_queue_maxsize=encoder_queue_maxsize,
        batch_encoding_size=video_batch_size if use_videos else 1,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec=dataset.VIDEO_CODEC, crf=video_crf, preset=video_preset),
    )

    total_frames = 0
    relabeled_episodes = 0
    try:
        for output_index, episode in enumerate(episodes):
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            validate_training_metadata(metadata)
            if int(metadata["fps"]) != dataset.DATASET_FPS:
                raise ValueError(
                    f"{episode.name} 的fps={metadata['fps']}，预期{dataset.DATASET_FPS}"
                )
            source_task = str(metadata.get("task", ""))
            if source_task != dataset.TASK:
                if not dataset.RELABEL_TASK_DURING_CONVERSION:
                    raise ValueError(f"{episode.name} 的任务文字与TASK不一致")
                relabeled_episodes += 1
                print(f"[任务重标] {episode.name}: 使用 constants.TASK")
            episode_frames = 0
            episode_started = last_progress = time.monotonic()
            schema_version = int(metadata.get("schema_version", 1))
            print(f"[开始 {output_index + 1}/{len(episodes)}] {episode.name}", flush=True)
            for row in read_jsonl(episode / "samples.jsonl"):
                state, action = episode_vectors(
                    row["observation.state"],
                    row["action"],
                    schema_version,
                )
                frame = {
                    "observation.state": state,
                    "action": action,
                    "task": dataset.TASK,
                }
                for camera in ("top", "wrist"):
                    image = cv2.imread(str(episode / row["images"][camera]), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (height, width, 3):
                        raise ValueError(f"{episode.name} 第{episode_frames}帧{camera}图像无效")
                    frame[f"observation.images.{camera}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
                episode_frames += 1
                now = time.monotonic()
                if now - last_progress >= 5:
                    print(f"[处理 {output_index + 1}/{len(episodes)}] {episode.name}: "
                          f"已送入{episode_frames}帧，"
                          f"{episode_frames / max(now - episode_started, 1e-9):.1f}帧/s", flush=True)
                    last_progress = now
            if episode_frames == 0:
                raise ValueError(f"{episode.name} 没有样本")
            print(f"[保存 {output_index + 1}/{len(episodes)}] {episode.name}: "
                  f"{episode_frames}帧，正在写入数据并编码视频…", flush=True)
            dataset.save_episode()
            total_frames += episode_frames
            elapsed = time.monotonic() - started
            rate = total_frames / max(elapsed, 1e-9)
            print(f"[转换 {output_index + 1}/{len(episodes)}] {episode.name}: {episode_frames}帧；"
                  f"累计{total_frames}帧，平均{rate:.1f}帧/s，已用{elapsed:.0f}s", flush=True)
    except BaseException:
        try:
            dataset.finalize()
        except Exception as cleanup_error:
            print(f"收尾失败：{cleanup_error}", file=sys.stderr)
        raise
    else:
        print("[收尾] 正在写入剩余元数据…", flush=True)
        dataset.finalize()

    print(
        f"LeRobot Dataset v3.0完成：{output_root}\n"
        f"episode={len(episodes)}, frames={total_frames}, fps={dataset.DATASET_FPS}, "
        f"task_relabeled={relabeled_episodes}"
    )
    return len(episodes), total_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=paths.RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=paths.LEROBOT_ROOT)
    parser.add_argument("--repo-id", default=dataset.DATASET_REPO_ID)
    parser.add_argument("--video-batch-size", type=int, default=1, help="本地LeRobot使用1；更大值自动回退到1")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-crf", type=float, default=30)
    parser.add_argument("--video-preset", default="fast")
    parser.add_argument("--streaming-encoding", action=argparse.BooleanOptionalAction, default=True,
                        help="默认边读边编码，省去中间PNG；--no-streaming-encoding恢复原模式")
    parser.add_argument("--encoder-threads", type=int, default=2, help="每路视频编码线程数")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=16, help="每路缓存帧数，满时等待不丢帧")
    args = parser.parse_args()
    try:
        convert_dataset(**vars(args))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"转换失败：{error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
