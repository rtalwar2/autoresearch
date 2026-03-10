# autoresearch — Diffusion Policy Hyperparameter Tuning

This is an experiment to have an LLM agent autonomously tune hyperparameters for a diffusion policy trained with LeRobot.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, log parsing, setup verification. Do not modify.
   - `train.py` — the file you modify. All tunable hyperparameters and config builder.
   - `train_config.json` — reference config (user's original). Read-only reference.
4. **Verify setup**: Run `uv run prepare.py` to verify lerobot is installed and the dataset is accessible.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed step budget of 5000 steps**. You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything in the hyperparameters section is fair game: vision backbone, UNet architecture (down_dims, n_groups, kernel_size), noise scheduler config, optimizer settings, learning rate, batch size, horizon, augmentation settings, etc.
- You may also modify the `build_config()` function if you need to add new config fields.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed step budget, dataset ID, input/output features, normalization mapping, and log parsing.
- Modify the input features or output features. These are pinned in `prepare.py`.
- Install new packages or add dependencies. Only use what's already available.
- Change `TRAINING_STEPS` (fixed at 5000 in `prepare.py`).

**The goal is simple: get the lowest train_loss.** Since the step budget is fixed, you don't need to worry about training length — it's always 5000 steps. Everything else is fair game: change the UNet architecture, the optimizer, the noise scheduler, the batch size, the vision backbone, the learning rate, the augmentation. The only constraint is the code runs without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful train_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
train_loss:       0.045678
total_seconds:    892.1
peak_vram_mb:     12345.0
steps:            5000
batch_size:       32
horizon:          8
lr:               0.0001
vision_backbone:  resnet18
down_dims:        [512, 1024, 2048]
scheduler:        DDIM
prediction_type:  epsilon
```

You can extract the key metric from the log file:

```
grep "^train_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	train_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. train_loss achieved (e.g. 0.045678) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	train_loss	memory_gb	status	description
a1b2c3d	0.045678	12.1	keep	baseline
b2c3d4e	0.042100	12.3	keep	increase LR to 0.0003
c3d4e5f	0.048000	12.1	discard	switch to DDPM scheduler
d4e5f6g	0.000000	0.0	crash	resnet50 backbone (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by editing the hyperparameters.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^train_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If train_loss improved (lower), you "advance" the branch, keeping the git commit
9. If train_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take roughly 10-25 minutes (depending on config). If a run exceeds 30 minutes, it is automatically killed and treated as a failure.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try different backbones, different UNet widths, different learning rates, different noise schedules, different batch sizes, different horizons, try removing augmentation, try adding weight decay, try different optimizers. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are some hyperparameter directions the agent can explore (non-exhaustive):

- **Learning rate**: try 3e-4, 5e-4, 1e-3, 3e-5
- **Batch size**: 16, 64, 128 (more samples per step vs. fewer steps in fixed budget)
- **Horizon**: 4, 12, 16, 32 (shorter = faster inference, longer = more context)
- **Vision backbone**: resnet18, resnet34, resnet50
- **UNet width**: smaller [256, 512, 1024] or larger [512, 1024, 2048, 4096]
- **Noise scheduler**: DDPM vs DDIM
- **Prediction type**: epsilon vs sample
- **num_train_timesteps**: 50, 100, 200, 1000
- **Kernel size**: 3, 5, 7
- **n_groups**: 4, 8, 16
- **diffusion_step_embed_dim**: 64, 128, 256
- **Spatial softmax keypoints**: 16, 32, 64
- **Weight decay**: 0, 1e-6, 1e-4, 1e-2
- **Grad clip norm**: 1.0, 5.0, 10.0, 50.0
- **Warmup steps**: 0, 100, 500, 1000
- **LR scheduler**: cosine, linear
- **Image augmentation**: disable entirely, change max transforms
- **crop_shape**: different crop sizes
- **use_amp**: True (mixed precision) for speed
- **n_obs_steps**: 1, 2, 3 (how many observation frames)
- **n_action_steps**: 1, 2, 4 (how many actions predicted per step)
- **Film scale modulation**: on/off
- **Group norm vs batch norm**: use_group_norm on/off
