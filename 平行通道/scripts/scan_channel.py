#!/usr/bin/env python3
"""平行通道掃描器 — 找出正在成型 / 剛完成（突破．跌破）的平行通道。

用法:
    python3 平行通道/scripts/scan_channel.py            # 預設港股+美股熱門名單
    python3 平行通道/scripts/scan_channel.py 0700.HK AAPL
    python3 平行通道/scripts/scan_channel.py --period 2y --order 6

邏輯:
 1. 擺動點 (pivot high / pivot low)，左右各 order 根棒確認
 2. 上軌 = 以 pivot high 做線性回歸；下軌 = 以 pivot low 做線性回歸
 3. 平行條件：兩軌斜率差距 <= 0.35 x 平均斜率 + 價格 0.1%/棒；上軌在
    所有 pivot low 之上、下軌在所有 pivot high 之下（通道內包覆率 >= 80%）
 4. 通道分類：上升 / 下降 / 橫行（斜率相對價格 < 0.05%/棒）
 5. 狀態：通道內 / 突破上軌 / 跌破下軌（用最近 close 判斷，超出半個 ATR 才算）
 6. 突破後結局（只報已突破者）：上升 / 橫行 / 下跌 — 用突破後 20 根棒判斷

輸出只作候選，實際畫線與結構必須在 TradingView 人手確認。
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 yfinance / numpy: pip install yfinance numpy ({exc})")

DEFAULT_TICKERS = [
    # 港股
    "0005.HK", "0016.HK", "0003.HK", "0700.HK", "0941.HK", "1299.HK",
    "0388.HK", "2318.HK", "0883.HK", "0981.HK", "1810.HK", "9988.HK",
    "3690.HK", "0066.HK", "0823.HK", "0960.HK", "1378.HK", "2628.HK",
    # 美股
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "GOOGL", "AMZN",
    "JPM", "XOM", "KO", "PG", "CAT", "BA", "MRNA",
]

ORDER = 6          # pivot 左右確認棒數
LOOKBACK = 180     # 只睇最近 N 根棒（約 8 個月日線）
MIN_TOUCH = 2      # 每條軌最少觸點
PARALLEL_TOL = 0.35
CONTAIN_RATIO = 0.80
BREAK_ATR = 0.5


def pivots(high: np.ndarray, low: np.ndarray, order: int):
    ph = [i for i in range(order, len(high) - order)
          if high[i] == max(high[i - order:i + order + 1])]
    pl = [i for i in range(order, len(low) - order)
          if low[i] == min(low[i - order:i + order + 1])]
    return ph, pl


def fit_line(idx, values):
    if len(idx) < 2:
        return None
    slope, intercept = np.polyfit(np.asarray(idx, dtype=float), np.asarray(values, dtype=float), 1)
    return float(slope), float(intercept)


def atr(high, low, close, n=14):
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    return float(np.mean(tr[-n:])) if len(tr) >= n else float(np.mean(tr))


def classify_outcome(closes, lows, highs, brk_i, level_fn, direction, a):
    """突破後 20 根棒：同一方向延伸 = 上升/下跌，來回震盪 = 橫行。"""
    if brk_i >= len(closes) - 5:
        return "突破太新（未夠棒數判斷）", None
    post = closes[brk_i: brk_i + 21]
    if len(post) < 6:
        return "突破太新（未夠棒數判斷）", None
    net = post[-1] - post[0]
    rng = post.max() - post.min()
    if rng < 2.0 * a:
        return "橫行（窄幅整理）", net
    if abs(net) < 0.8 * a:
        return "橫行（來回震盪）", net
    return ("上升（順突破方向延伸）" if net > 0 else "下跌（反向）"), net


def analyse(ticker: str, period: str, order: int):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or len(df) < LOOKBACK // 2:
        return None
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df = df.tail(LOOKBACK)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    if len(close) < 60:
        return None
    a = atr(high, low, close)
    last = float(close[-1])

    ph, pl = pivots(high, low, order)
    if len(ph) < MIN_TOUCH or len(pl) < MIN_TOUCH:
        return None

    up = fit_line(ph, high[ph])
    dn = fit_line(pl, low[pl])
    if not up or not dn:
        return None
    su, bu = up
    sd, bd = dn

    n = len(close)
    scale = last * 0.0001                      # 0.01%/棒 作為斜率比較基準
    if abs(su - sd) > PARALLEL_TOL * max(abs(su), abs(sd), scale) + scale * 10:
        return None

    x = np.arange(n)
    upper = su * x + bu
    lower = sd * x + bd
    if (upper <= lower).any():
        return None

    # 包覆率：多少根棒完全在通道內
    inside = ((close <= upper + a) & (close >= lower - a)).sum() / n
    if inside < CONTAIN_RATIO:
        return None

    width = float(np.mean(upper - lower))
    if width < 1.5 * a:                        # 太窄冇交易意義
        return None

    slope_pct = (su + sd) / 2 / last * 100     # 每棒 % 斜率
    if abs(slope_pct) < 0.05:
        kind = "橫行通道"
    elif slope_pct > 0:
        kind = "上升通道"
    else:
        kind = "下降通道"

    up_now, lo_now = upper[-1], lower[-1]
    status, brk_i, direction = "通道內", None, None
    if last > up_now + BREAK_ATR * a:
        status, direction = "突破上軌", +1
    elif last < lo_now - BREAK_ATR * a:
        status, direction = "跌破下軌", -1
    if direction:
        beyond = close > (upper + BREAK_ATR * a) if direction > 0 else close < (lower - BREAK_ATR * a)
        # 取「最近一次、仍然持續」的越軌起點：第一次穿軌多數係假突破
        brk_i = None
        if beyond[-1]:
            i = len(beyond) - 1
            while i > 0 and beyond[i - 1]:
                i -= 1
            brk_i = i

    outcome, net = (None, None)
    if brk_i is not None:
        outcome, net = classify_outcome(
            close, low, high, brk_i,
            (lambda i, d=direction: upper[i] if d > 0 else lower[i]), direction, a)

    return {
        "ticker": ticker, "kind": kind, "status": status,
        "slope_pct": round(slope_pct, 3), "inside": round(float(inside), 2),
        "width_atr": round(width / a, 1), "last": round(last, 2),
        "touches": f"{len(ph)}H/{len(pl)}L",
        "break_date": str(df.index[brk_i].date()) if brk_i is not None else "-",
        "outcome": outcome, "net": None if net is None else round(float(net), 2),
    }


def main(argv):
    args = [x for x in argv[1:] if not x.startswith("-")]
    period = "1y"
    if "--period" in argv:
        period = argv[argv.index("--period") + 1]
    order = ORDER
    if "--order" in argv:
        order = int(argv[argv.index("--order") + 1])
    tickers = [t for t in args if t not in (period, str(order))] or DEFAULT_TICKERS

    rows = []
    for t in tickers:
        try:
            r = analyse(t, period, order)
        except Exception as exc:                            # noqa: BLE001
            print(f"  ! {t}: {type(exc).__name__}")
            continue
        if r:
            rows.append(r)

    if not rows:
        print("冇發現符合條件嘅平行通道（放寬 --order 或延長 --period 再試）")
        return 0

    rows.sort(key=lambda r: (r["status"] == "通道內", r["inside"]))
    print(f"\n平行通道掃描 · {len(tickers)} 隻 · period={period} order={order}\n")
    print(f"{'代號':<10}{'類型':<8}{'狀態':<10}{'斜率%/棒':>9}{'包覆':>6}{'寬/ATR':>7}  {'觸點':<8}{'突破日':<12}結局")
    print("-" * 108)
    for r in rows:
        out = r["outcome"] or ""
        if r["net"] is not None:
            out += f" ({r['net']:+})"
        print(f"{r['ticker']:<10}{r['kind']:<8}{r['status']:<10}{r['slope_pct']:>9}"
              f"{r['inside']:>6}{r['width_atr']:>7}  {r['touches']:<8}{r['break_date']:<12}{out}")
    print("\n※ 通道掃描只作候選；線位、突破真偽同結構（BOS/CHoCH）必須在 TradingView 人手確認。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
