"""Generate MRNA trend-continuation chart for homework project (real data)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os

# CJK font for Chinese labels
for fp in ['/System/Library/Fonts/Hiragino Sans GB.ttc',
           '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
           '/System/Library/Fonts/STHeiti Medium.ttc']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images')

# ── MRNA real data (6/22 – 8/19 2026) ──
raw = [
    # date, open, high, low, close
    ('06-22', 64.07, 65.51, 57.50, 59.35),
    ('06-23', 59.05, 63.98, 58.34, 61.00),
    ('06-24', 61.17, 62.75, 59.85, 60.42),
    ('06-25', 60.33, 61.86, 57.91, 59.75),
    ('06-26', 59.47, 69.29, 59.23, 67.27),
    ('06-29', 67.95, 69.90, 66.30, 69.70),
    ('06-30', 69.83, 73.28, 69.22, 70.03),
    ('07-01', 70.04, 73.91, 68.87, 72.50),
    ('07-02', 73.71, 81.40, 73.68, 79.76),
    ('07-06', 78.55, 85.60, 77.50, 81.80),
    ('07-07', 82.10, 82.90, 77.41, 79.77),
    ('07-08', 77.34, 78.30, 73.34, 73.80),
    ('07-09', 73.80, 76.84, 73.14, 76.56),
    ('07-10', 76.72, 76.72, 67.31, 68.27),
    ('07-13', 66.33, 69.55, 65.54, 67.01),
    ('07-14', 66.67, 67.89, 65.80, 67.44),
    ('07-15', 67.79, 68.49, 65.95, 68.28),
    ('07-16', 67.07, 67.42, 62.80, 63.15),
    ('07-17', 61.68, 63.28, 61.06, 61.82),
    ('07-20', 61.46, 62.42, 59.30, 59.49),
    ('07-21', 59.51, 60.83, 58.61, 59.66),
    ('07-22', 59.52, 59.86, 57.74, 58.07),
    ('07-23', 57.20, 58.43, 56.28, 57.02),
    ('07-24', 57.26, 57.46, 53.85, 54.07),
    ('07-27', 54.75, 57.07, 54.39, 55.63),
    ('07-28', 54.83, 55.85, 53.35, 55.81),
    ('07-29', 55.04, 56.25, 54.16, 54.49),
    ('07-30', 55.97, 58.08, 54.52, 57.92),
    ('07-31', 57.85, 59.49, 54.71, 54.82),
    ('08-03', 54.92, 58.00, 52.66, 55.14),
    ('08-04', 56.84, 58.34, 55.88, 56.99),
    ('08-05', 58.00, 59.45, 55.58, 56.26),
    ('08-06', 57.54, 59.00, 53.71, 53.86),
    ('08-07', 54.79, 59.42, 54.62, 59.17),
    ('08-10', 59.75, 60.77, 58.41, 59.81),
    ('08-11', 60.21, 61.40, 58.60, 60.57),
    ('08-12', 60.58, 64.26, 59.29, 63.67),
    ('08-13', 63.80, 65.33, 62.33, 63.65),
    ('08-14', 62.89, 63.78, 60.82, 63.32),
    ('08-17', 64.78, 65.53, 62.91, 64.46),
    ('08-18', 63.04, 64.46, 62.13, 62.96),
]
dates = [r[0] for r in raw]
opens = np.array([r[1] for r in raw])
highs = np.array([r[2] for r in raw])
lows = np.array([r[3] for r in raw])
closes = np.array([r[4] for r in raw])
n = len(raw)

# ── structure levels ──
prev_peak = 85.60          # 前推進浪頂 (7/6 high) = Stop Earn #1
wave1_start = 59.23        # 推進浪起點 (6/26 low)
retrace_low = 52.66        # 回調低點 (8/3 low) = 假突破底部 reference
# Fib 1.618 extension: C + (B-A)*1.618
fib_1618 = retrace_low + (prev_peak - wave1_start) * 1.618  # 52.66 + 26.37*1.618 ≈ 95.33

fig, ax = plt.subplots(figsize=(11, 6), facecolor='#1a1f2e')

# K lines
for i in range(n):
    color = '#22c55e' if closes[i] >= opens[i] else '#ef4444'
    ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8, zorder=2)
    body_lo = min(opens[i], closes[i])
    body_hi = max(opens[i], closes[i])
    ax.add_patch(mpatches.Rectangle(
        (i - 0.3, body_lo), 0.6, max(body_hi - body_lo, 0.4),
        facecolor=color, edgecolor=color, zorder=3
    ))

# Levels
ax.axhline(y=prev_peak, color='#8b5cf6', alpha=0.5, linewidth=1.2, linestyle='--', zorder=1)
ax.text(n - 0.5, prev_peak, f'  前推進浪頂 (Support) $85.60', color='#8b5cf6',
        fontsize=8.5, va='bottom', ha='right', fontweight='bold')
ax.text(n - 0.5, prev_peak + 4, 'Stop Earn #1', color='#22c55e',
        fontsize=8, va='bottom', ha='right', fontfamily='monospace')

ax.axhline(y=fib_1618, color='#22c55e', alpha=0.45, linewidth=1, linestyle='--', zorder=1)
ax.text(n - 0.5, fib_1618, f'  回調浪 1.618 ≈ $95.3', color='#22c55e',
        fontsize=8, va='bottom', ha='right')
ax.text(n - 0.5, fib_1618 + 5, 'Stop Earn #2', color='#22c55e',
        fontsize=8, va='bottom', ha='right', fontfamily='monospace')

ax.axhline(y=retrace_low, color='#ef4444', alpha=0.5, linewidth=1.2, linestyle='--', zorder=1)
ax.text(n - 0.5, retrace_low, f'  回調低點 $52.66（假突破底部 → SL 下方）', color='#ef4444',
        fontsize=8, va='bottom', ha='right')

# Annotations
ax.annotate('推進浪 ①', xy=(10, 81.5), xytext=(5, 90),
            fontsize=9, color='#22d3ee', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=1.2))
ax.annotate('回調（-38%）', xy=(25, 56), xytext=(19, 47),
            fontsize=9, color='#f97316', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f97316', lw=1.2))
ax.annotate('推進浪 ②\n(8/19 爆升突破前頂 $176.7)', xy=(40, 65), xytext=(34, 74),
            fontsize=9, color='#22d3ee', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=1.2))

# wave 1 line
ax.plot([5, 10], [wave1_start, prev_peak], color='#64748b', linestyle=':', linewidth=1, alpha=0.6)

ax.set_xlim(-1, n + 1)
ax.set_ylim(45, 105)
ax.set_title('MRNA · Moderna · 1D · 2026-06-22 → 08-19  |  順勢交易案例', color='#e2e8f0',
             fontsize=12, fontweight='bold', loc='left', pad=8)
ax.set_xticks(range(n))
ax.set_xticklabels(dates, rotation=45, fontsize=6.5, color='#64748b')
ax.tick_params(colors='#64748b', labelsize=7)
ax.set_facecolor('#1a1f2e')
for spine in ax.spines.values():
    spine.set_color('#334155')
ax.yaxis.grid(color='#334155', alpha=0.3, linewidth=0.5)

fig.text(0.02, 0.01, 'MRNA real data · yfinance · 順勢交易筆記', color='#475569', fontsize=7)
plt.tight_layout(pad=0.5)
out = os.path.join(OUTPUT_DIR, 'mrna-trend-continuation.png')
fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#1a1f2e')
plt.close(fig)
print(f'Generated: {out}')
print(f'fib_1618 = {fib_1618:.2f} | prev_peak = {prev_peak} | retrace_low = {retrace_low}')
