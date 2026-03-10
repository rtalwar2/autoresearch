"""
Fixed preparation and evaluation for diffusion policy HP tuning.
Contains constants, log parsing, and setup verification.
DO NOT MODIFY — this is the fixed harness.

Usage:
    uv run prepare.py          # verify setup
"""

import os
import sys
import re
import subprocess
import threading

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TRAINING_STEPS = 5000         # fixed step budget per experiment
DATASET_REPO_ID = "ramen-noodels/delta_xyz_final_rgb"
TIMEOUT_SECONDS = 1800        # 30 min hard safety timeout
LOG_FREQ = 100                # logging frequency (steps)

# Fixed input/output features — DO NOT CHANGE
INPUT_FEATURES = {
    "observation.images.wrist_image": {
        "type": "VISUAL",
        "shape": [3, 240, 320]
    },
    "observation.state": {
        "type": "STATE",
        "shape": [1]
    }
}

OUTPUT_FEATURES = {
    "action": {
        "type": "ACTION",
        "shape": [3]
    }
}

NORMALIZATION_MAPPING = {
    "VISUAL": "MEAN_STD",
    "STATE": "MIN_MAX",
    "ACTION": "MIN_MAX"
}

# ---------------------------------------------------------------------------
# Log parsing (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def parse_train_loss(log_file):
    """
    Parse lerobot training log and return the average train loss
    over the last 5 log entries. Returns None if no loss found.

    LeRobot log format: "step:5K smpl:160K ... loss:0.123 ..."
    """
    with open(log_file) as f:
        text = f.read()
    # Match "loss:0.123" pattern from MetricsTracker output
    losses = re.findall(r'\bloss:(\d+\.?\d*)', text)
    if not losses:
        return None
    recent = [float(x) for x in losses[-5:]]
    return sum(recent) / len(recent)


def parse_peak_vram_mb(log_file):
    """Try to extract peak VRAM from a gpu_monitor log file. Returns 0.0 on failure."""
    try:
        with open(log_file) as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return max(values) if values else 0.0
    except (FileNotFoundError, ValueError):
        return 0.0

# ---------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------

def start_gpu_monitor(log_path, interval=2):
    """Start a background thread that polls nvidia-smi and logs peak VRAM."""
    stop_event = threading.Event()

    def _monitor():
        with open(log_path, "w") as f:
            while not stop_event.is_set():
                try:
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    mb = r.stdout.strip().split('\n')[0]
                    f.write(mb + '\n')
                    f.flush()
                except Exception:
                    pass
                stop_event.wait(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return stop_event


# ---------------------------------------------------------------------------
# Setup verification
# ---------------------------------------------------------------------------

def verify_setup():
    """Verify lerobot is importable and dataset is accessible."""
    print("Checking setup...")

    # Check lerobot
    try:
        import lerobot
        print(f"  lerobot: OK")
    except ImportError:
        print("  ERROR: lerobot not importable. Install it or run: uv sync")
        sys.exit(1)

    # Check dataset accessibility
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        print(f"  Loading dataset metadata for {DATASET_REPO_ID}...")
        meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
        print(f"  Dataset: {meta.num_episodes} episodes, {meta.num_frames} frames")
    except Exception as e:
        print(f"  ERROR: Could not access dataset: {e}")
        sys.exit(1)

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
        else:
            print("  WARNING: No CUDA GPU detected")
    except Exception:
        print("  WARNING: Could not check GPU")

    print()
    print("Setup OK! Ready to train.")


if __name__ == "__main__":
    verify_setup()
