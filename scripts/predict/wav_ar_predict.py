"""AR model prediction from a WAV file.

Usage:
    python wav_ar_predict.py audio.wav --n_pred 500 --order 16
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR prediction from WAV file")
    p.add_argument("file", type=str, help="Path to .wav file")
    p.add_argument("--n_pred", type=int, default=500,
                   help="Number of samples to predict (default: 500)")
    p.add_argument("--order", type=int, default=16,
                   help="AR order p (default: 16)")
    p.add_argument("--n_windows", type=int, default=50,
                   help="Number of windows for variance check (default: 50)")
    p.add_argument("--context", type=int, default=1000,
                   help="Samples of real data shown before prediction (default: 1000)")
    p.add_argument("--out", type=str, default="fig/ar_prediction.png",
                   help="Output image path (default: fig/ar_prediction.png)")
    return p.parse_args()


def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # take first channel if stereo
    return data.astype(np.float64), sr


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------

def windowed_variance(x: np.ndarray, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = n // n_windows
    centers = np.arange(n_windows) * w + w // 2
    variances = np.array([x[i * w:(i + 1) * w].var() for i in range(n_windows)])
    return centers, variances


# ---------------------------------------------------------------------------
# Hilbert envelope normalization
# ---------------------------------------------------------------------------

def normalize_envelope(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide signal by its smoothed Hilbert envelope."""
    envelope = np.abs(hilbert(x))
    smooth_len = max(len(x) // 100, 1)
    envelope = uniform_filter1d(envelope, size=smooth_len)
    envelope = np.maximum(envelope, 1e-8)
    return x / envelope, envelope


# ---------------------------------------------------------------------------
# AR parameter estimation
# ---------------------------------------------------------------------------

def fit_yule_walker(x: np.ndarray, p: int) -> np.ndarray:
    """Moment-based: solve Yule-Walker equations from sample autocorrelation."""
    x = x - x.mean()
    n = len(x)
    r = np.array([x[:n - k] @ x[k:] / n for k in range(p + 1)])
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    return np.linalg.solve(R, r[1:])


def fit_ols(x: np.ndarray, p: int) -> np.ndarray:
    """Conditional least squares: OLS on lagged design matrix."""
    n = len(x)
    # columns: x[t-1], x[t-2], ..., x[t-p]
    X = np.column_stack([x[p - 1 - i:n - 1 - i] for i in range(p)])
    y = x[p:]
    phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return phi


def fit_mle(x: np.ndarray, p: int) -> np.ndarray:
    """Exact MLE via statsmodels AutoReg (handles initial conditions)."""
    from statsmodels.tsa.ar_model import AutoReg
    res = AutoReg(x, lags=p, trend="n", old_names=False).fit(method="mle")
    return res.params  # shape (p,) with trend='n'


# ---------------------------------------------------------------------------
# AIC
# ---------------------------------------------------------------------------

def compute_aic(x: np.ndarray, phi: np.ndarray) -> float:
    """AIC = n*log(RSS/n) + 2*(p+1)  (p AR coefs + 1 noise variance param)."""
    p = len(phi)
    n = len(x)
    X = np.column_stack([x[p - 1 - i:n - 1 - i] for i in range(p)])
    residuals = x[p:] - X @ phi
    n_eff = len(residuals)
    rss = residuals @ residuals
    return n_eff * np.log(rss / n_eff) + 2 * (p + 1)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_ar(x: np.ndarray, phi: np.ndarray, n_pred: int) -> np.ndarray:
    """Recursive one-step-ahead prediction."""
    p = len(phi)
    buf = list(x[-p:])
    out = []
    for _ in range(n_pred):
        # phi[0]*x[t-1] + phi[1]*x[t-2] + ...
        v = float(phi @ np.array(buf[-p:][::-1]))
        out.append(v)
        buf.append(v)
    return np.array(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    x, sr = load_wav(args.file)
    p, n_pred = args.order, args.n_pred
    print(f"Loaded : {args.file}  samples={len(x)}  sr={sr} Hz  duration={len(x)/sr:.2f}s")

    t = np.arange(len(x)) / sr

    # --- stationarity before normalization ---
    c_raw, v_raw = windowed_variance(x, args.n_windows)
    t_c = c_raw / sr

    # --- Hilbert envelope normalization ---
    x_norm, envelope = normalize_envelope(x)
    c_norm, v_norm = windowed_variance(x_norm, args.n_windows)

    # --- fit AR ---
    print(f"Fitting AR({p}) ...")
    phi_yw  = fit_yule_walker(x_norm, p)
    phi_ols = fit_ols(x_norm, p)
    phi_mle = fit_mle(x_norm, p)
    aic_yw  = compute_aic(x_norm, phi_yw)
    aic_ols = compute_aic(x_norm, phi_ols)
    aic_mle = compute_aic(x_norm, phi_mle)
    for name, phi, aic in [
        ("Yule-Walker", phi_yw,  aic_yw),
        ("OLS",         phi_ols, aic_ols),
        ("MLE",         phi_mle, aic_mle),
    ]:
        print(f"  {name:12s}  AIC={aic:.2f}  phi[:4]={np.round(phi[:4], 4)}")

    # --- predict ---
    pred_yw  = predict_ar(x_norm, phi_yw,  n_pred)
    pred_ols = predict_ar(x_norm, phi_ols, n_pred)
    pred_mle = predict_ar(x_norm, phi_mle, n_pred)

    ctx = min(args.context, len(x))
    t_ctx  = np.arange(len(x) - ctx, len(x)) / sr
    t_pred = np.arange(len(x), len(x) + n_pred) / sr

    # --- figure ---
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.5,
                           height_ratios=[2.5, 1.2, 2.5, 1.2, 3])
    fig.suptitle(f"AR({p}) Prediction  —  {Path(args.file).name}", fontsize=13, fontweight="bold")

    # 1. original signal
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, x, lw=0.35, color="steelblue", alpha=0.8)
    ax1.set_title("Original signal", fontsize=10)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    # 2. windowed variance — original
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_c, v_raw, color="tomato", lw=1.2, marker="o", ms=3)
    ax2.set_title("Windowed variance — original  (non-stationary: variance changes over time)", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Variance")

    # 3. envelope-normalized signal
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, x_norm, lw=0.35, color="steelblue", alpha=0.75, label="normalized")
    ax3.set_title("Envelope-normalized signal  (divided by smoothed Hilbert envelope)", fontsize=10)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")

    # 4. windowed variance — normalized
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t_c, v_norm, color="seagreen", lw=1.2, marker="o", ms=3)
    ax4.set_title("Windowed variance — normalized  (should be flatter)", fontsize=10)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Variance")

    # 5. prediction comparison
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t_ctx, x_norm[-ctx:], color="black", lw=0.8, label=f"actual (last {ctx} samples)")
    ax5.plot(t_pred, pred_yw,  color="tomato",     lw=1.1, alpha=0.9, label=f"Yule-Walker  AIC={aic_yw:.1f}")
    ax5.plot(t_pred, pred_ols, color="seagreen",   lw=1.1, alpha=0.9, label=f"OLS          AIC={aic_ols:.1f}",  ls="--")
    ax5.plot(t_pred, pred_mle, color="darkorange", lw=1.1, alpha=0.9, label=f"MLE          AIC={aic_mle:.1f}",  ls=":")
    ax5.axvline(len(x) / sr, color="gray", lw=0.8, ls="--")
    ax5.set_title(f"AR({p}) prediction  ({n_pred} samples = {n_pred/sr:.3f}s)", fontsize=10)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Amplitude (normalized)")
    ax5.legend(fontsize=8, loc="upper left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved  : {out}")
    plt.show()


if __name__ == "__main__":
    main()
