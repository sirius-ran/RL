"""
Train an inverse prior actor for BC-prior residual RL.

The prior predicts normalized opening delta:
    prior_action = clip((u[t+1] - u[t]) / PRIOR_MAX_DELTA, -1, 1)

Set INVERSE_PRIOR_LABEL_MODE=model_inverse to train on one-step model-inverted
targets instead of historical operator deltas.  Use rollout_inverse to choose
labels by a short closed-loop rollout, which is slower but accounts for delay.

It can then be used with:
    ACTION_MODE=bc_prior_residual
    PRIOR_OUTPUT_MODE=delta
    BC_PRIOR_ACTOR_PATH=models/actors/inverse_prior_actor.pth
"""

from pathlib import Path
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train_controller_ddpg import (
    Actor,
    BoilerGymEnvPhysics,
    ShadowAugmentor,
    build_demo_state,
    model_d,
    params,
    shadow_model,
    ROOT,
    COL_U,
    COL_Y,
    COL_LOAD,
    CTRL_HORIZON,
    SHADOW_HORIZON,
    device,
)


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def one_step_model_inverse_delta(env, t):
    env.reset_at(t)
    curr_u = float(env.curr_u)
    y_dist_next = float(env.model_d.predict(env._build_input_d())[0])
    current_load = float(env.df.iloc[env.step_idx][COL_LOAD])
    t_load = env._T_load(current_load)
    a = np.exp(-env.dt / t_load)
    b = 1.0 - a
    x1_prev, x2_prev, x3_prev = env.control_state

    deltas = np.arange(-INVERSE_MAX_DELTA, INVERSE_MAX_DELTA + 1e-9,
                       max(INVERSE_GRID_STEP, 1e-6), dtype=np.float32)
    best_delta = 0.0
    best_score = float("inf")
    best_pred_y = None

    for delta in deltas:
        next_u = float(np.clip(curr_u + float(delta), 0.0, 100.0))
        actual_delta = next_u - curr_u
        delayed_queue = list(env.u_delay_queue)
        delayed_queue.append(next_u)
        max_len = env.delay + 1
        if len(delayed_queue) > max_len:
            delayed_queue = delayed_queue[-max_len:]
        u_delayed = delayed_queue[0] if len(delayed_queue) > env.delay else env.u_ss

        next_x1 = a * x1_prev + b * env.K * (u_delayed - env.u_ss)
        next_x2 = a * x2_prev + b * next_x1
        next_x3 = a * x3_prev + b * next_x2
        pred_y = y_dist_next + next_x3
        track_error = abs(pred_y - env.target_sp)
        score = track_error + INVERSE_LAMBDA_U * abs(actual_delta)
        if score < best_score:
            best_score = score
            best_delta = actual_delta
            best_pred_y = pred_y

    return float(np.clip(best_delta, -INVERSE_MAX_DELTA, INVERSE_MAX_DELTA)), float(best_pred_y)


def rollout_inverse_delta(env, t):
    base_state = env.reset_at(t).copy()
    curr_u = float(env.curr_u)

    deltas = np.arange(-INVERSE_MAX_DELTA, INVERSE_MAX_DELTA + 1e-9,
                       max(INVERSE_GRID_STEP, 1e-6), dtype=np.float32)
    best_delta = 0.0
    best_score = float("inf")
    best_rollout_mae = None

    for delta in deltas:
        env.reset_at(t)
        next_u = float(np.clip(curr_u + float(delta), 0.0, 100.0))
        actual_delta = next_u - curr_u
        abs_errors = []
        done = False
        for h in range(ROLLOUT_HORIZON):
            _, _, done, info = env.step(next_u)
            abs_errors.append(abs(float(info["y_rl"]) - env.target_sp))
            if done:
                break
        if not abs_errors:
            continue

        weights = np.power(ROLLOUT_DECAY, np.arange(len(abs_errors), dtype=np.float32))
        weights = weights / max(float(weights.sum()), 1e-6)
        weighted_mae = float(np.sum(np.asarray(abs_errors, dtype=np.float32) * weights))
        final_error = float(abs_errors[-1])
        score = (weighted_mae
                 + ROLLOUT_FINAL_WEIGHT * final_error
                 + INVERSE_LAMBDA_U * abs(actual_delta))
        if score < best_score:
            best_score = score
            best_delta = actual_delta
            best_rollout_mae = weighted_mae

    env.reset_at(t)
    _ = base_state
    return float(np.clip(best_delta, -INVERSE_MAX_DELTA, INVERSE_MAX_DELTA)), best_rollout_mae


TRAIN_DATA_PATH = Path(os.environ.get("TRAIN_DATA_PATH", ROOT / "data/raw/RHT0121quan.csv"))
OUT_PATH = Path(os.environ.get(
    "INVERSE_PRIOR_OUT", ROOT / "models/actors/inverse_prior_actor.pth"))
RUN_TAG = os.environ.get("RUN_TAG", "inverse_prior").strip() or "inverse_prior"

PRIOR_MAX_DELTA = env_float("PRIOR_MAX_DELTA", 10.0)
LABEL_MODE = os.environ.get("INVERSE_PRIOR_LABEL_MODE", "history").strip().lower()
INVERSE_MAX_DELTA = env_float("INV_PRIOR_INVERSE_MAX_DELTA", PRIOR_MAX_DELTA)
INVERSE_GRID_STEP = env_float("INV_PRIOR_INVERSE_GRID_STEP", 0.5)
INVERSE_LAMBDA_U = env_float("INV_PRIOR_INVERSE_LAMBDA_U", 0.03)
ROLLOUT_HORIZON = env_int("INV_PRIOR_ROLLOUT_HORIZON", 12)
ROLLOUT_DECAY = env_float("INV_PRIOR_ROLLOUT_DECAY", 0.95)
ROLLOUT_FINAL_WEIGHT = env_float("INV_PRIOR_ROLLOUT_FINAL_WEIGHT", 0.5)
MAX_ABS_DEMO_DELTA = env_float("INV_PRIOR_MAX_ABS_DEMO_DELTA", 15.0)
QUALITY_MAE_THRESHOLD = env_float("INV_PRIOR_QUALITY_MAE_THRESHOLD", 8.0)
QUALITY_WINDOW = env_int("INV_PRIOR_QUALITY_WINDOW", 3)
VAL_SPLIT = env_float("INV_PRIOR_VAL_SPLIT", 0.2)
EPOCHS = env_int("INV_PRIOR_EPOCHS", 80)
BATCH_SIZE = env_int("INV_PRIOR_BATCH_SIZE", 256)
LR = env_float("INV_PRIOR_LR", 1e-4)
MOVE_DEADZONE = env_float("INV_PRIOR_MOVE_DEADZONE", 1.0) / PRIOR_MAX_DELTA
DIR_WEIGHT = env_float("INV_PRIOR_DIR_WEIGHT", 0.2)
STAY_WEIGHT = env_float("INV_PRIOR_STAY_WEIGHT", 0.2)
SAMPLE_STRIDE = env_int("INV_PRIOR_SAMPLE_STRIDE", 1)
MAX_SAMPLES = env_int("INV_PRIOR_MAX_SAMPLES", 0)
SEED = env_int("SEED", 42)
SHOW_PLOTS = env_bool("SHOW_PLOTS", False)

if LABEL_MODE not in {"history", "model_inverse", "rollout_inverse"}:
    raise ValueError(
        "INVERSE_PRIOR_LABEL_MODE must be 'history', 'model_inverse', or 'rollout_inverse'."
    )


def build_dataset():
    env = BoilerGymEnvPhysics(TRAIN_DATA_PATH, model_d, params,
                              shadow_predictor=shadow_model,
                              noise_std=0.0)
    shadow_aug = ShadowAugmentor(shadow_model, n_lags=10,
                                 horizon=SHADOW_HORIZON,
                                 target=env.target_sp)
    states, targets, raw_deltas = [], [], []
    inverse_pred_errors = []
    skipped_quality = 0
    skipped_delta = 0
    t_start = env.n_lags
    t_end = len(env.df) - max(CTRL_HORIZON + 1, SHADOW_HORIZON + 1) - 1

    for n_seen, t in enumerate(range(t_start, t_end, max(SAMPLE_STRIDE, 1)), start=1):
        if MAX_SAMPLES > 0 and len(states) >= MAX_SAMPLES:
            break
        if LABEL_MODE == "rollout_inverse" and n_seen % 250 == 0:
            print(f"[dataset] scanned={n_seen:,}, samples={len(states):,}", flush=True)
        window = env.df[COL_Y].iloc[max(0, t - QUALITY_WINDOW):
                                    t + QUALITY_WINDOW + 1].values
        local_mae = float(np.mean(np.abs(window - env.target_sp)))
        if local_mae > QUALITY_MAE_THRESHOLD:
            skipped_quality += 1
            continue

        s = build_demo_state(env, shadow_aug, t, target_sp=env.target_sp)
        curr_u = float(env.df[COL_U].iloc[t])
        if LABEL_MODE == "model_inverse":
            delta_u, pred_y = one_step_model_inverse_delta(env, t)
            inverse_pred_errors.append(pred_y - env.target_sp)
        elif LABEL_MODE == "rollout_inverse":
            delta_u, rollout_mae = rollout_inverse_delta(env, t)
            inverse_pred_errors.append(rollout_mae)
        else:
            next_u = float(env.df[COL_U].iloc[t + 1])
            delta_u = next_u - curr_u
            if abs(delta_u) > MAX_ABS_DEMO_DELTA:
                skipped_delta += 1
                continue
        target = float(np.clip(delta_u / PRIOR_MAX_DELTA, -1.0, 1.0))
        states.append(s)
        targets.append([target])
        raw_deltas.append(delta_u)

    total = max(t_end - t_start, 1)
    print(f"[dataset] samples={len(states):,}")
    print(f"[dataset] label_mode={LABEL_MODE}")
    print(f"[dataset] quality skipped={skipped_quality:,} ({skipped_quality / total * 100:.1f}%)")
    print(f"[dataset] delta skipped={skipped_delta:,} ({skipped_delta / total * 100:.1f}%)")
    if inverse_pred_errors and LABEL_MODE == "model_inverse":
        inv_err = np.asarray(inverse_pred_errors, dtype=np.float32)
        print(f"[dataset] inverse one-step MAE={np.mean(np.abs(inv_err)):.3f}C")
    elif inverse_pred_errors and LABEL_MODE == "rollout_inverse":
        inv_err = np.asarray(inverse_pred_errors, dtype=np.float32)
        print(f"[dataset] inverse rollout weighted MAE={np.mean(inv_err):.3f}C")
    if not states:
        raise RuntimeError("No inverse-prior samples were built; relax filters.")

    return (np.asarray(states, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(raw_deltas, dtype=np.float32))


def inverse_prior_loss(pred, target):
    mag_loss = F.smooth_l1_loss(pred, target)
    move_mask = torch.abs(target) >= MOVE_DEADZONE
    if move_mask.any():
        dir_loss = F.softplus(-torch.sign(target[move_mask]) * pred[move_mask]).mean()
    else:
        dir_loss = torch.zeros((), device=pred.device)
    if (~move_mask).any():
        stay_loss = F.smooth_l1_loss(pred[~move_mask], torch.zeros_like(pred[~move_mask]))
    else:
        stay_loss = torch.zeros((), device=pred.device)
    return mag_loss + DIR_WEIGHT * dir_loss + STAY_WEIGHT * stay_loss


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(x).to(device)).cpu().numpy()
    err_delta = (pred[:, 0] - y[:, 0]) * PRIOR_MAX_DELTA
    true_delta = y[:, 0] * PRIOR_MAX_DELTA
    move_mask = np.abs(true_delta) >= MOVE_DEADZONE * PRIOR_MAX_DELTA
    if np.any(move_mask):
        dir_acc = float(np.mean(np.sign(pred[:, 0][move_mask]) == np.sign(y[:, 0][move_mask])) * 100.0)
    else:
        dir_acc = 0.0
    return {
        "mae_delta": float(np.mean(np.abs(err_delta))),
        "rmse_delta": float(np.sqrt(np.mean(err_delta ** 2))),
        "dir_acc": dir_acc,
        "pred": pred,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"device={device}")
    print(f"TRAIN_DATA_PATH={TRAIN_DATA_PATH}")
    print(f"INVERSE_PRIOR_LABEL_MODE={LABEL_MODE}")
    print(f"PRIOR_MAX_DELTA={PRIOR_MAX_DELTA}")
    if LABEL_MODE in {"model_inverse", "rollout_inverse"}:
        print(f"INV_PRIOR_INVERSE_MAX_DELTA={INVERSE_MAX_DELTA}")
        print(f"INV_PRIOR_INVERSE_GRID_STEP={INVERSE_GRID_STEP}")
        print(f"INV_PRIOR_INVERSE_LAMBDA_U={INVERSE_LAMBDA_U}")
    if LABEL_MODE == "rollout_inverse":
        print(f"INV_PRIOR_ROLLOUT_HORIZON={ROLLOUT_HORIZON}")
        print(f"INV_PRIOR_ROLLOUT_DECAY={ROLLOUT_DECAY}")
        print(f"INV_PRIOR_ROLLOUT_FINAL_WEIGHT={ROLLOUT_FINAL_WEIGHT}")
    print(f"QUALITY_MAE_THRESHOLD={QUALITY_MAE_THRESHOLD}")
    print(f"INV_PRIOR_SAMPLE_STRIDE={SAMPLE_STRIDE}")
    print(f"INV_PRIOR_MAX_SAMPLES={MAX_SAMPLES}")

    x, y, raw_deltas = build_dataset()
    split = int(len(x) * (1.0 - VAL_SPLIT))
    split = min(max(split, 1), len(x) - 1)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    print(f"[split] train={len(x_train):,}, val={len(x_val):,}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = Actor(x.shape[1], CTRL_HORIZON).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = inverse_prior_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        val_stats = evaluate(model, x_val, y_val)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_mae_delta": val_stats["mae_delta"],
            "val_rmse_delta": val_stats["rmse_delta"],
            "val_dir_acc": val_stats["dir_acc"],
        }
        history.append(row)
        print(f"Ep {epoch:03d} | train_loss={row['train_loss']:.4f} | "
              f"val_delta_MAE={row['val_mae_delta']:.3f}% | "
              f"dir_acc={row['val_dir_acc']:.1f}%")

        if row["val_mae_delta"] < best_val:
            best_val = row["val_mae_delta"]
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  [best] saved {OUT_PATH}")

    hist_path = ROOT / "results/evaluation" / f"inverse_prior_history_{RUN_TAG}.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")

    best_state = torch.load(OUT_PATH, map_location=device)
    model.load_state_dict(best_state)
    val_stats = evaluate(model, x_val, y_val)
    pred_delta = val_stats["pred"][:, 0] * PRIOR_MAX_DELTA
    true_delta = y_val[:, 0] * PRIOR_MAX_DELTA

    fig_path = ROOT / "results/figures" / f"inverse_prior_report_{RUN_TAG}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(true_delta[:1000], label="demo delta", alpha=0.8)
    axes[0].plot(pred_delta[:1000], label="prior pred delta", alpha=0.8)
    axes[0].set_ylabel("opening delta (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot([h["val_mae_delta"] for h in history], label="val delta MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("delta MAE (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Inverse prior training ({RUN_TAG})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[saved] actor={OUT_PATH}")
    print(f"[saved] history={hist_path}")
    print(f"[saved] figure={fig_path}")
    print(f"[best] val_delta_MAE={best_val:.3f}%")


if __name__ == "__main__":
    main()
