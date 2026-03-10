"""
Diffusion policy hyperparameter tuning script. Single-GPU.
This is the ONLY file you modify.

Usage: uv run train.py
"""

import os
import sys
import json
import time
import subprocess

from prepare import (
    TRAINING_STEPS, DATASET_REPO_ID, INPUT_FEATURES, OUTPUT_FEATURES,
    NORMALIZATION_MAPPING, TIMEOUT_SECONDS, LOG_FREQ,
    parse_train_loss, parse_peak_vram_mb, start_gpu_monitor,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Policy architecture
HORIZON = 8
N_OBS_STEPS = 2
N_ACTION_STEPS = 2
VISION_BACKBONE = "resnet18"
CROP_SHAPE = [216, 288]
CROP_IS_RANDOM = True
PRETRAINED_BACKBONE_WEIGHTS = None
USE_GROUP_NORM = True
SPATIAL_SOFTMAX_NUM_KEYPOINTS = 32
USE_SEPARATE_RGB_ENCODER_PER_CAMERA = True

# UNet architecture
DOWN_DIMS = [512, 1024, 2048]
KERNEL_SIZE = 5
N_GROUPS = 8
DIFFUSION_STEP_EMBED_DIM = 128
USE_FILM_SCALE_MODULATION = True

# Noise scheduler
NOISE_SCHEDULER_TYPE = "DDIM"
NUM_TRAIN_TIMESTEPS = 100
BETA_SCHEDULE = "squaredcos_cap_v2"
BETA_START = 0.0001
BETA_END = 0.02
PREDICTION_TYPE = "epsilon"
CLIP_SAMPLE = True
CLIP_SAMPLE_RANGE = 1.0

# Optimizer
LR = 1e-4
BETAS = [0.95, 0.999]
EPS = 1e-8
WEIGHT_DECAY = 1e-6
GRAD_CLIP_NORM = 10.0

# LR scheduler
SCHEDULER_NAME = "cosine"
WARMUP_STEPS = 500

# Training
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 2025

# Image augmentation
IMAGE_TRANSFORMS_ENABLE = True
IMAGE_TRANSFORMS_MAX_NUM = 3

# ---------------------------------------------------------------------------
# Config builder (edit if you add new HP fields)
# ---------------------------------------------------------------------------

def build_config():
    """Build the full lerobot training config dict from the HP constants above."""
    drop_n_last = HORIZON - 1

    return {
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "root": None,
            "episodes": None,
            "image_transforms": {
                "enable": IMAGE_TRANSFORMS_ENABLE,
                "max_num_transforms": IMAGE_TRANSFORMS_MAX_NUM,
                "random_order": True,
                "tfs": {
                    "brightness": {"weight": 1.0, "type": "ColorJitter",
                                   "kwargs": {"brightness": [0.5, 1.5]}},
                    "contrast": {"weight": 1.0, "type": "ColorJitter",
                                 "kwargs": {"contrast": [0.5, 1.5]}},
                    "saturation": {"weight": 1.0, "type": "ColorJitter",
                                   "kwargs": {"saturation": [0.5, 1.5]}},
                    "hue": {"weight": 1.0, "type": "ColorJitter",
                            "kwargs": {"hue": [-0.1, 0.1]}},
                    "sharpness": {"weight": 1.0, "type": "SharpnessJitter",
                                  "kwargs": {"sharpness": [0.5, 1.5]}},
                },
            },
            "revision": None,
            "use_imagenet_stats": False,
            "video_backend": "torchcodec",
        },
        "env": None,
        "policy": {
            "type": "diffusion",
            "n_obs_steps": N_OBS_STEPS,
            "normalization_mapping": NORMALIZATION_MAPPING,
            "input_features": INPUT_FEATURES,
            "output_features": OUTPUT_FEATURES,
            "device": "cuda",
            "use_amp": False,
            "horizon": HORIZON,
            "n_action_steps": N_ACTION_STEPS,
            "drop_n_last_frames": drop_n_last,
            "vision_backbone": VISION_BACKBONE,
            "crop_shape": CROP_SHAPE,
            "crop_is_random": CROP_IS_RANDOM,
            "pretrained_backbone_weights": PRETRAINED_BACKBONE_WEIGHTS,
            "use_group_norm": USE_GROUP_NORM,
            "spatial_softmax_num_keypoints": SPATIAL_SOFTMAX_NUM_KEYPOINTS,
            "use_separate_rgb_encoder_per_camera": USE_SEPARATE_RGB_ENCODER_PER_CAMERA,
            "audio_backbone": None,
            "down_dims": DOWN_DIMS,
            "kernel_size": KERNEL_SIZE,
            "n_groups": N_GROUPS,
            "diffusion_step_embed_dim": DIFFUSION_STEP_EMBED_DIM,
            "use_film_scale_modulation": USE_FILM_SCALE_MODULATION,
            "noise_scheduler_type": NOISE_SCHEDULER_TYPE,
            "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
            "beta_schedule": BETA_SCHEDULE,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "prediction_type": PREDICTION_TYPE,
            "clip_sample": CLIP_SAMPLE,
            "clip_sample_range": CLIP_SAMPLE_RANGE,
            "num_inference_steps": None,
            "do_mask_loss_for_padding": False,
            "optimizer_lr": LR,
            "optimizer_betas": BETAS,
            "optimizer_eps": EPS,
            "optimizer_weight_decay": WEIGHT_DECAY,
            "scheduler_name": SCHEDULER_NAME,
            "scheduler_warmup_steps": WARMUP_STEPS,
        },
        "job_name": "autoresearch_diffusion",
        "resume": False,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "batch_size": BATCH_SIZE,
        "steps": TRAINING_STEPS,
        "eval_freq": 0,
        "log_freq": LOG_FREQ,
        "save_checkpoint": False,
        "save_freq": TRAINING_STEPS + 1,
        "use_policy_training_preset": True,
        "optimizer": {
            "type": "adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "betas": BETAS,
            "eps": EPS,
        },
        "scheduler": {
            "type": "diffuser",
            "num_warmup_steps": WARMUP_STEPS,
            "name": SCHEDULER_NAME,
        },
        "eval": {
            "n_episodes": 0,
            "batch_size": 1,
            "use_async_envs": False,
        },
        "wandb": {
            "enable": False,
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    # Build and write config
    config = build_config()
    config_path = os.path.join(script_dir, "_autoresearch_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Dataset:  {DATASET_REPO_ID}")
    print(f"Steps:    {TRAINING_STEPS}")
    print(f"Batch:    {BATCH_SIZE}")
    print(f"Horizon:  {HORIZON}")
    print(f"LR:       {LR}")
    print(f"Backbone: {VISION_BACKBONE}")
    print(f"Down dims:{DOWN_DIMS}")
    print(f"Timeout:  {TIMEOUT_SECONDS}s")
    print()

    # Start GPU memory monitor
    vram_log = os.path.join(script_dir, "_gpu_vram.log")
    stop_monitor = start_gpu_monitor(vram_log)

    # Run lerobot training as subprocess
    train_log = os.path.join(script_dir, "_train_output.log")
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
    ]

    exit_code = -1
    try:
        with open(train_log, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
            )
            exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Training exceeded {TIMEOUT_SECONDS}s, killed.")
        exit_code = -1

    # Stop GPU monitor
    stop_monitor.set()
    time.sleep(0.5)

    t_end = time.time()
    total_seconds = t_end - t_start

    # Parse results
    train_loss = parse_train_loss(train_log) if exit_code == 0 else None
    peak_vram_mb = parse_peak_vram_mb(vram_log)

    # Print summary
    print()
    print("---")
    if train_loss is not None:
        print(f"train_loss:       {train_loss:.6f}")
    else:
        print(f"train_loss:       FAILED")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"steps:            {TRAINING_STEPS}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"horizon:          {HORIZON}")
    print(f"lr:               {LR}")
    print(f"vision_backbone:  {VISION_BACKBONE}")
    print(f"down_dims:        {DOWN_DIMS}")
    print(f"scheduler:        {NOISE_SCHEDULER_TYPE}")
    print(f"prediction_type:  {PREDICTION_TYPE}")
