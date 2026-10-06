import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# UCM 验证集普通口径 mean（来自 cup-deploy-assets/plot_curves.py）
orig = [41.11, 40.71, 43.11, 46.05, 47.33, 47.27, 47.48, 47.91, 49.28, 49.50,
        49.84, 49.35, 51.02, 51.05, 50.60, 51.53, 51.51, 51.27, 51.05, 52.02,
        50.99, 52.28, 51.34, 52.02, 51.64, 51.71, 51.48, 51.54, 51.61, 52.61,
        52.24, 52.40, 52.51, 52.84, 51.79, 52.35, 51.74, 52.30, 52.25, 51.96,
        51.63, 52.82, 51.49, 51.31, 51.65, 51.48, 51.42, 51.36, 51.37, 51.37]

v2 = [39.87, 41.62, 43.92, 45.30, 46.76, 47.19, 48.37, 48.56, 48.80, 50.37,
      52.14, 52.14, 53.56, 53.70, 54.92, 54.76, 55.27, 54.64, 54.78, 55.80,
      55.87, 55.34, 56.43, 56.65, 55.58, 55.93, 55.71, 56.78, 55.53, 56.27,
      56.73, 55.61, 56.04, 54.68, 54.91, 54.82, 54.29, 55.13, 54.67, 54.82,
      54.87, 54.29, 54.86, 55.37, 54.77, 54.70, 54.77, 54.75, 54.64, 54.64]

cold = [41.80, 45.12, 47.93, 51.69, 51.69, 52.81, 53.48, 54.14, 54.06, 54.77,
        54.44, 54.74, 55.16, 54.62, 55.29, 55.12, 55.35, 54.65, 55.56, 54.37,
        55.74, 55.65, 54.44, 55.01, 54.97, 54.72, 54.21, 53.73, None, 54.51,
        54.32, None, 53.75, None, 54.60, None, 54.59, None, None, None,
        55.69, None, 55.03, None, 54.90, None, None, 55.14, None, 55.69]

x = np.arange(1, 51)
cold = np.array(cold, dtype=float)  # None -> NaN，画图自动断线

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                     "axes.labelsize": 12, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.plot(x, orig, color="#0072B2", linestyle="-",  lw=1.6, label="Original CUP (CCB)")
ax.plot(x, v2,   color="#009E73", linestyle="-.", lw=1.6, label="Ours (SFT-init)")
ax.plot(x, cold, color="#D55E00", linestyle="--", lw=1.6, label="Ours (Cold-start GRPO)")

for val, c in [(orig[-1], "#0072B2"), (v2[-1], "#009E73"), (cold[-1], "#D55E00")]:
    ax.axhline(val, color=c, lw=0.5, ls=":", alpha=0.5)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation mR (%)")
ax.set_xlim(1, 50)
ax.set_ylim(38, 60)
ax.grid(True, linestyle=":", lw=0.4, color="#888888", alpha=0.5)
ax.legend(loc="lower right", frameon=False, fontsize=10)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("/Users/botlv/CUP-code/figs/main_curves.png", dpi=300)
plt.savefig("/Users/botlv/CUP-code/figs/main_curves.pdf")
print("saved: main_curves.png / main_curves.pdf")
print(f"终点值  orig={orig[-1]}  v2={v2[-1]}  cold={cold[-1]}")
