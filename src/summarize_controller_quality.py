from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "evaluation"
OUT_PATH = EVAL_DIR / "controller_quality_comparison.csv"

SETPOINT = 606.0
SAT_LOW = 5.0
SAT_HIGH = 95.0
DELTA_DEADZONE = 0.5


def version_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "full_controller_timeseries_"
    if stem == "full_controller_timeseries":
        return "pureRL"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def tracking_metrics(diff: pd.Series) -> dict:
    values = diff.to_numpy(dtype=float)
    abs_values = np.abs(values)
    return {
        "mae": float(abs_values.mean()),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "within_2c_pct": float((abs_values <= 2.0).mean() * 100.0),
        "within_5c_pct": float((abs_values <= 5.0).mean() * 100.0),
    }


def action_metrics(u: pd.Series) -> dict:
    values = u.to_numpy(dtype=float)
    if len(values) <= 1:
        du = np.array([], dtype=float)
    else:
        du = np.diff(values)
    abs_du = np.abs(du)

    move_mask = abs_du >= DELTA_DEADZONE
    move_sign = np.sign(du[move_mask])
    if len(move_sign) <= 1:
        flip_rate = 0.0
    else:
        flip_rate = float((move_sign[1:] * move_sign[:-1] < 0).mean() * 100.0)

    return {
        "u_mean": float(values.mean()),
        "u_min": float(values.min()),
        "u_max": float(values.max()),
        "u_range": float(values.max() - values.min()),
        "mean_abs_delta_u": float(abs_du.mean()) if len(abs_du) else 0.0,
        "p95_abs_delta_u": float(np.percentile(abs_du, 95)) if len(abs_du) else 0.0,
        "max_abs_delta_u": float(abs_du.max()) if len(abs_du) else 0.0,
        "total_abs_delta_u": float(abs_du.sum()) if len(abs_du) else 0.0,
        "move_rate_pct": float(move_mask.mean() * 100.0) if len(move_mask) else 0.0,
        "direction_flip_rate_pct": flip_rate,
        "saturation_rate_pct": float(((values <= SAT_LOW) | (values >= SAT_HIGH)).mean() * 100.0),
    }


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    version = version_from_path(path)

    if "diff_rl" in df:
        diff_rl = df["diff_rl"]
    else:
        diff_rl = df["y_rl"] - SETPOINT
    if "diff_origin" in df:
        diff_origin = df["diff_origin"]
    else:
        diff_origin = df["y_origin"] - SETPOINT

    row = {
        "version": version,
        "n": int(len(df)),
        "source": str(path.relative_to(ROOT)),
    }
    row.update(tracking_metrics(diff_rl))
    row.update({f"origin_{k}": v for k, v in tracking_metrics(diff_origin).items()})
    row.update({f"rl_{k}": v for k, v in action_metrics(df["u_rl"]).items()})
    row.update({f"origin_{k}": v for k, v in action_metrics(df["u_origin"]).items()})
    return row


def main():
    paths = sorted(EVAL_DIR.glob("full_controller_timeseries*.csv"))
    rows = [summarize_file(path) for path in paths]
    out = pd.DataFrame(rows)

    metric_cols = [c for c in out.columns if c not in {"version", "n", "source"}]
    out[metric_cols] = out[metric_cols].round(4)
    out = out.sort_values(["mae", "rmse"], ascending=[True, True])
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    display_cols = [
        "version",
        "mae",
        "rmse",
        "within_2c_pct",
        "within_5c_pct",
        "rl_mean_abs_delta_u",
        "rl_p95_abs_delta_u",
        "rl_saturation_rate_pct",
        "rl_direction_flip_rate_pct",
    ]
    print(out[display_cols].head(20).to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
