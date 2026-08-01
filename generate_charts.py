"""Generate breakout trading chart images for homework project."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIB_COLORS = {
    '0': '#ef4444', '0.236': '#f97316', '0.382': '#eab308',
    '0.5': '#22c55e', '0.618': '#14b8a6', '0.786': '#3b82f6', '1': '#8b5cf6'
}

CHARTS = [
    {
        'file': '1378-hk.png', 'title': '中國宏橋集團 1378 · 1D · HKEX',
        'trend': 'bull', 'bos_count': 6, 'fib_levels': {'0': 24.04, '0.236': 20.16, '0.382': 18.94, '0.5': 17.95, '0.618': 16.96, '0.786': 15.55, '1': 13.76},
        'price_range': (6, 26)
    },
    {
        'file': 'msft.png', 'title': 'Microsoft MSFT · 1D · NASDAQ',
        'trend': 'bull_long', 'bos_count': 8, 'fib_levels': {'0': 446, '0.236': 380, '0.382': 340, '0.5': 310, '0.618': 280, '0.786': 230, '1': 134},
        'price_range': (100, 480)
    },
    {
        'file': '0941-hk.png', 'title': '中國移動 0941 · 1D · HKEX',
        'trend': 'bull', 'bos_count': 7, 'fib_levels': {'0': 52.95, '0.236': 49.68, '0.382': 47.46, '0.5': 45.68, '0.618': 43.90, '0.786': 41.37, '1': 39.12},
        'price_range': (20, 55)
    },
    {
        'file': '7287-t.png', 'title': '日本精機 7287 · 1D · TSE',
        'trend': 'bull', 'bos_count': 5, 'fib_levels': {'0': 2391, '0.236': 2100, '0.382': 1900, '0.5': 1758, '0.618': 1600, '0.786': 1400, '1': 1255},
        'price_range': (1100, 2500)
    },
    {
        'file': 'ddd.png', 'title': '3D Systems DDD · 1D · NYSE',
        'trend': 'boom_bust', 'bos_count': 4, 'fib_levels': {'0': 84.45, '0.236': 71.16, '0.382': 62.94, '0.5': 56.30, '0.618': 49.66, '0.786': 40.20, '1': 28.15},
        'price_range': (25, 100)
    },
    {
        'file': 'sony.png', 'title': 'Sony SONY · 1D · NYSE',
        'trend': 'breakout', 'bos_count': 5, 'fib_levels': {'0': 15.94, '0.236': 14.24, '0.382': 13.19, '0.5': 12.34, '0.618': 11.49, '0.786': 10.28, '1': 8.74},
        'price_range': (7, 18)
    },
    {
        'file': 'msi.png', 'title': 'Motorola Solutions MSI · 1D · NYSE',
        'trend': 'bull', 'bos_count': 5, 'fib_levels': {'0': 246.15, '0.236': 231.51, '0.382': 222.45, '0.5': 215.13, '0.618': 207.81, '0.786': 197.38, '1': 184.10},
        'price_range': (170, 260)
    },
    {
        'file': '0005-hk.png', 'title': '滙豐控股 0005 · 1D · HKEX',
        'trend': 'recovery', 'bos_count': 6, 'fib_levels': {'0': 147.7, '0.236': 119.4, '0.382': 101.9, '0.5': 87.80, '0.618': 73.65, '0.786': 53.55, '1': 27.92},
        'price_range': (20, 170)
    },
    {
        'file': '8252-t.png', 'title': '丸井集團 8252 · 1D · TSE',
        'trend': 'consolidation', 'bos_count': 4, 'fib_levels': {'0': 3108, '0.236': 2850, '0.382': 2691, '0.5': 2562, '0.618': 2433, '0.786': 2250, '1': 2016},
        'price_range': (1900, 3200)
    },
    {
        'file': '0052-hk.png', 'title': '大快活集團 0052 · 1D · HKEX',
        'trend': 'parabolic', 'bos_count': 5, 'fib_levels': {'0': 28.14, '0.236': 25.14, '0.382': 23.28, '0.5': 21.78, '0.618': 20.30, '0.786': 18.50, '1': 15.43},
        'price_range': (10, 30)
    },
    {
        'file': '0003-hk.png', 'title': '香港中華煤氣 0003 · 1D · HKEX',
        'trend': 'bull_then_bear', 'bos_count': 5, 'fib_levels': {'0': 13.00, '0.236': 12.11, '0.382': 11.56, '0.5': 11.12, '0.618': 10.67, '0.786': 10.04, '1': 9.24},
        'price_range': (7, 14)
    },
    {
        'file': 'ba.png', 'title': 'Boeing BA · 1D · NYSE',
        'trend': 'bull', 'bos_count': 7, 'fib_levels': {'0': 366, '0.236': 310, '0.382': 275, '0.5': 245, '0.618': 215, '0.786': 175, '1': 130},
        'price_range': (120, 380)
    },
]


def generate_price_data(n=80, trend='bull', price_range=(10, 100)):
    np.random.seed(42)
    lo, hi = price_range
    mid = (lo + hi) / 2

    if trend == 'bull':
        base = np.linspace(lo, hi * 0.95, n)
        noise = np.cumsum(np.random.randn(n) * (hi - lo) * 0.008)
        prices = base + noise
    elif trend == 'bull_long':
        base = np.linspace(lo, hi * 0.9, n)
        noise = np.cumsum(np.random.randn(n) * (hi - lo) * 0.006)
        prices = base + noise
    elif trend == 'boom_bust':
        peak = int(n * 0.55)
        up = np.linspace(lo, hi, peak)
        down = np.linspace(hi * 0.95, lo * 1.3, n - peak)
        prices = np.concatenate([up, down])
        prices += np.cumsum(np.random.randn(n) * (hi - lo) * 0.01)
    elif trend == 'breakout':
        consol = int(n * 0.4)
        flat = np.ones(consol) * lo * 1.1 + np.random.randn(consol) * (hi - lo) * 0.02
        breakout = np.linspace(lo * 1.1, hi, n - consol)
        prices = np.concatenate([flat, breakout])
        prices += np.cumsum(np.random.randn(n) * (hi - lo) * 0.005)
    elif trend == 'recovery':
        base = np.linspace(lo, hi * 0.92, n)
        noise = np.cumsum(np.random.randn(n) * (hi - lo) * 0.007)
        prices = base + noise
    elif trend == 'consolidation':
        first = np.linspace(lo, hi * 0.85, int(n * 0.6))
        flat = np.ones(n - int(n * 0.6)) * hi * 0.85 + np.random.randn(n - int(n * 0.6)) * (hi - lo) * 0.03
        prices = np.concatenate([first, flat])
    elif trend == 'parabolic':
        t = np.linspace(0, 1, n)
        prices = lo + (hi - lo) * (t ** 2.5)
        prices += np.cumsum(np.random.randn(n) * (hi - lo) * 0.005)
    elif trend == 'bull_then_bear':
        peak = int(n * 0.7)
        up = np.linspace(lo, hi, peak)
        down = np.linspace(hi * 0.98, lo * 1.05, n - peak)
        prices = np.concatenate([up, down])
        prices += np.cumsum(np.random.randn(n) * (hi - lo) * 0.006)
    else:
        prices = np.linspace(lo, hi, n)

    return np.clip(prices, lo * 0.9, hi * 1.05)


def draw_chart(config):
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10, 5.5), gridspec_kw={'height_ratios': [4, 1]},
        facecolor='#1a1f2e'
    )

    lo, hi = config['price_range']
    n = 80
    closes = generate_price_data(n, config['trend'], config['price_range'])
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * (hi - lo) * 0.01
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * (hi - lo) * 0.01
    volumes = np.abs(np.random.randn(n)) * 1e6 + 5e5

    fib = config['fib_levels']
    fib_keys = ['0', '0.236', '0.382', '0.5', '0.618', '0.786', '1']
    for i, key in enumerate(fib_keys):
        level = fib[key]
        color = FIB_COLORS[key]
        ax_price.axhline(y=level, color=color, alpha=0.25, linewidth=8, zorder=0)
        ax_price.text(0.99, level, f'  {key} ({level})', transform=ax_price.get_yaxis_transform(),
                      fontsize=7, color=color, va='center', ha='right', fontfamily='monospace')

    for i in range(n):
        color = '#22c55e' if closes[i] >= opens[i] else '#ef4444'
        ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8, zorder=2)
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        ax_price.add_patch(mpatches.Rectangle(
            (i - 0.3, body_lo), 0.6, max(body_hi - body_lo, (hi - lo) * 0.002),
            facecolor=color, edgecolor=color, zorder=3
        ))

    bos_indices = np.linspace(10, n - 10, config['bos_count'], dtype=int)
    for idx in bos_indices:
        ax_price.annotate('BOS', (idx, highs[idx]), fontsize=7, color='#22d3ee',
                          fontweight='bold', ha='center', va='bottom')

    trend_x = [0, n - 1]
    trend_y = [closes[0], closes[-1]]
    ax_price.plot(trend_x, trend_y, color='#64748b', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    ax_price.set_xlim(-1, n)
    ax_price.set_ylim(lo * 0.95, hi * 1.02)
    ax_price.set_title(config['title'], color='#e2e8f0', fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_price.tick_params(colors='#64748b', labelsize=7)
    ax_price.set_facecolor('#1a1f2e')
    for spine in ax_price.spines.values():
        spine.set_color('#334155')

    vol_colors = ['#22c55e' if closes[i] >= opens[i] else '#ef4444' for i in range(n)]
    ax_vol.bar(range(n), volumes, color=vol_colors, alpha=0.5, width=0.8)
    ax_vol.set_facecolor('#1a1f2e')
    ax_vol.tick_params(colors='#64748b', labelsize=6)
    ax_vol.set_xlim(-1, n)
    for spine in ax_vol.spines.values():
        spine.set_color('#334155')

    fig.text(0.02, 0.01, 'TradingView · LuxAlgo SMC', color='#475569', fontsize=7)
    plt.tight_layout(pad=0.5)
    out_path = os.path.join(OUTPUT_DIR, config['file'])
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#1a1f2e')
    plt.close(fig)
    print(f'Generated: {out_path}')


if __name__ == '__main__':
    for chart in CHARTS:
        np.random.seed(hash(chart['file']) % 2**31)
        draw_chart(chart)
    print('Done!')
