import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def simulate_ar(phi: list[float], n_steps: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = len(phi)
    x = np.zeros(n_steps)
    noise = rng.normal(0, 1, n_steps)
    for t in range(p, n_steps):
        x[t] = sum(phi[i] * x[t - i - 1] for i in range(p)) + noise[t]
    return x


def characteristic_roots(phi: list[float]) -> np.ndarray:
    p = len(phi)
    companion = np.zeros((p, p))
    companion[0, :] = phi
    companion[1:, :-1] = np.eye(p - 1)
    return np.linalg.eigvals(companion)


r = 0.9
omega6 = 2 * np.pi / 6
omega12 = 2 * np.pi / 12

CASES: dict[str, list[tuple[str, list[float]]]] = {
    "AR(1)": [
        ("phi=0.9  slow decay", [0.9]),
        ("phi=-0.9  alternating", [-0.9]),
        ("phi=1.0  random walk", [1.0]),
        ("phi=0.3  weak persistence", [0.3]),
    ],
    "AR(2)": [
        ("real roots (0.6, 0.3)", [0.9, -0.18]),
        (f"complex roots  period≈6  r={r}", [2 * r * np.cos(omega6), -(r**2)]),
        (f"complex roots  period≈12  r={r}", [2 * r * np.cos(omega12), -(r**2)]),
        ("unstable", [1.2, -0.2]),
    ],
    "AR(3)": [
        ("slow decay", [0.5, 0.2, 0.1]),
        ("complex oscillation", [0.8, -0.5, 0.2]),
        ("seasonal-like  period≈4", [0.0, 0.0, 0.9]),
        ("mixed", [0.6, 0.3, -0.4]),
    ],
}

COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]
N_CASES = 4
N_GROUPS = len(CASES)

fig = plt.figure(figsize=(16, N_GROUPS * N_CASES * 1.8))
fig.suptitle("AR(n) Model Simulation", fontsize=15, fontweight="bold", y=1.001)

outer = gridspec.GridSpec(N_GROUPS, 1, figure=fig, hspace=0.55)

for group_idx, (ar_label, cases) in enumerate(CASES.items()):
    inner = gridspec.GridSpecFromSubplotSpec(
        N_CASES, 2,
        subplot_spec=outer[group_idx],
        hspace=0.15,
        wspace=0.08,
        width_ratios=[5, 1],
    )

    ax_roots = fig.add_subplot(inner[:, 1])
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_roots.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)
    ax_roots.axhline(0, color="gray", lw=0.4)
    ax_roots.axvline(0, color="gray", lw=0.4)
    ax_roots.set_aspect("equal")
    ax_roots.set_title("Char. roots", fontsize=8)
    ax_roots.tick_params(labelsize=7)
    ax_roots.set_xlabel("Re", fontsize=7)
    ax_roots.set_ylabel("Im", fontsize=7)

    for case_idx, (label, phi) in enumerate(cases):
        ax = fig.add_subplot(inner[case_idx, 0])
        ts = simulate_ar(phi, n_steps=300)
        ax.plot(ts, color=COLORS[case_idx], lw=0.85, alpha=0.9)
        ax.set_ylabel(label, fontsize=7.5, labelpad=4)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 300)

        if case_idx == 0:
            ax.set_title(ar_label, fontsize=11, fontweight="bold", loc="left", pad=6)
        if case_idx < N_CASES - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("t", fontsize=8)

        roots = characteristic_roots(phi)
        ax_roots.scatter(
            roots.real, roots.imag,
            color=COLORS[case_idx], s=35, zorder=5,
            label=f"case {case_idx + 1}",
        )

    ax_roots.legend(fontsize=6, loc="lower right", framealpha=0.7)

plt.savefig("fig/ar_simulation.png", dpi=150, bbox_inches="tight")
plt.show()
