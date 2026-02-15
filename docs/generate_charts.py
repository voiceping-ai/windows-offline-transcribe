"""Generate benchmark chart images for README."""
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")

# Benchmark data (sorted by inference time ascending)
models = [
    ("moonshine-tiny",    "sherpa-onnx",  383,   57.5, "English"),
    ("sensevoice-small",  "sherpa-onnx",  443,   49.6, "zh/en/ja/ko/yue"),
    ("moonshine-base",    "sherpa-onnx",  653,   33.7, "English"),
    ("parakeet-tdt-v2",   "sherpa-onnx",  984,   22.4, "English"),
    ("zipformer-20m",     "sherpa-onnx",  1312,  16.8, "English"),
    ("whisper-tiny",      "whisper.cpp",  1811,  12.1, "99 languages"),
    ("omnilingual-300m",  "sherpa-onnx",  2059,  0,    "1600+ languages"),
    ("whisper-base",      "whisper.cpp",  3907,  5.6,  "99 languages"),
    ("qwen3-asr-0.6b",   "qwen-asr",     13632, 1.6,  "52 languages"),
    ("whisper-small",     "whisper.cpp",  18942, 1.2,  "99 languages"),
]

HIGHLIGHT = {"parakeet-tdt-v2", "qwen3-asr-0.6b"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "images")

# Color palette by engine
ENGINE_COLORS = {
    "whisper.cpp": "#4A90D9",
    "sherpa-onnx": "#50C878",
    "qwen-asr":    "#E8833A",
}


def make_inference_chart():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    names = [m[0] for m in models]
    times = [m[2] for m in models]
    engines = [m[1] for m in models]
    colors = [ENGINE_COLORS.get(e, "#888") for e in engines]
    edge_colors = ["#222" if n in HIGHLIGHT else "none" for n in names]
    linewidths = [2 if n in HIGHLIGHT else 0 for n in names]

    y_pos = range(len(names))
    bars = ax.barh(y_pos, times, color=colors, edgecolor=edge_colors,
                   linewidth=linewidths, height=0.65)

    # Add time labels
    for bar, t, name in zip(bars, times, names):
        label = f" {t:,} ms"
        if name in HIGHLIGHT:
            label += "  \u2605"
        ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, fontweight="bold" if name in HIGHLIGHT else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Inference Time (ms) — lower is better", fontsize=11)
    ax.set_title("Inference Speed: 11s JFK Audio (i5-1035G1, CPU-only)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(times) * 1.25)
    ax.axvline(x=11000, color="#CC3333", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(11000 + 200, len(names) - 0.5, "real-time\n(11s audio)",
            color="#CC3333", fontsize=8, va="center")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=e) for e, c in ENGINE_COLORS.items()]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "benchmark-inference.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Created benchmark-inference.png")


def make_throughput_chart():
    # Filter out models with 0 words/s
    data = [(m[0], m[1], m[3]) for m in models if m[3] > 0]
    # Sort by throughput descending
    data.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [d[0] for d in data]
    wps = [d[2] for d in data]
    engines = [d[1] for d in data]
    colors = [ENGINE_COLORS.get(e, "#888") for e in engines]
    edge_colors = ["#222" if n in HIGHLIGHT else "none" for n in names]
    linewidths = [2 if n in HIGHLIGHT else 0 for n in names]

    y_pos = range(len(names))
    bars = ax.barh(y_pos, wps, color=colors, edgecolor=edge_colors,
                   linewidth=linewidths, height=0.65)

    for bar, w, name in zip(bars, wps, names):
        label = f" {w:.1f} w/s"
        if name in HIGHLIGHT:
            label += "  \u2605"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, fontweight="bold" if name in HIGHLIGHT else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Words per Second — higher is better", fontsize=11)
    ax.set_title("Throughput: 11s JFK Audio (i5-1035G1, CPU-only)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(wps) * 1.2)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=e) for e, c in ENGINE_COLORS.items()]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "benchmark-throughput.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Created benchmark-throughput.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_inference_chart()
    make_throughput_chart()
    print("Done.")
