#!/usr/bin/env python3
"""出圖：平行通道「突破／跌破＝趨勢終結」之後的三種結局（真實日線）。

每個案例都由 scan 邏輯（pivot 回歸）在**前半段**擬合通道，之後觀察價格離開
通道後實際走去邊 — 三個案例刻意揀三種唔同結局：

  1. BA      上升通道 → 突破上軌 → 之後下跌  （突破唔等於繼續升）
  2. KO      上升通道 → 跌破下軌 → 之後上升  （跌破唔等於一定要跌）
  3. 0027.HK 下降通道 → 突破上軌 → 之後橫行  （突破唔等於方向確立）

用法: python3 平行通道/scripts/make_charts.py
輸出: images/channel/*.png
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from scan_channel import atr, fit_line, pivots

BG = "#1a1f2e"
UP = "#22c55e"
DN = "#ef4444"
LINE = "#22d3ee"
BREAK = "#facc15"
TXT = "#e2e8f0"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "images", "channel")

plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC", "Hiragino Sans GB"]

CASES = [
    {"ticker": "BA", "file": "ba-channel-breakout-down.png", "fit_end": 180, "order": 5,
     "label": "波音 BA · 上升通道突破上軌 → 之後下跌",
     "outcome": "突破＝趨勢終結，唔等於繼續升 → 結果跌"},
    {"ticker": "KO", "file": "ko-channel-breakdown-up.png", "fit_end": 180, "order": 5,
     "label": "可口可樂 KO · 上升通道跌破下軌 → 之後上升",
     "outcome": "跌破＝趨勢終結，唔等於要跌 → 結果升"},
    {"ticker": "0027.HK", "file": "0027-hk-channel-break-flat.png", "fit_end": 150, "order": 5,
     "label": "銀河娛樂 0027 · 下降通道突破上軌 → 之後橫行",
     "outcome": "突破＝趨勢終結，方向未定 → 結果橫行"},
]


def load(ticker: str):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def draw(case):
    df = load(case["ticker"])
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    n = len(c)
    a = atr(h, l, c)
    fe, order = case["fit_end"], case["order"]

    ph, pl = pivots(h[:fe], l[:fe], order)
    su, bu = fit_line(ph, h[ph])
    sd, bd = fit_line(pl, l[pl])

    x = np.arange(n)
    upper, lower = su * x + bu, sd * x + bd

    beyond_up, beyond_dn = c > upper + 0.5 * a, c < lower - 0.5 * a
    brk, d = fe - 5, 1
    for i in range(fe - 5, n):
        if beyond_up[i] or beyond_dn[i]:
            brk, d = i, (1 if beyond_up[i] else -1)
            break

    net = (c[-1] - c[brk]) / a
    outcome = "橫行" if abs(net) < 1.5 else ("上升" if net > 0 else "下跌")

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(11.5, 6.4),
                                  gridspec_kw={"height_ratios": [4.2, 1]}, facecolor=BG)

    ax.plot(x, upper, color=LINE, lw=1.4, ls="--", zorder=3, label="通道上軌")
    ax.plot(x, lower, color=LINE, lw=1.4, ls="--", zorder=3, label="通道下軌")
    ax.fill_between(x, lower, upper, color=LINE, alpha=0.07, zorder=0)

    for i in range(n):
        col = UP if c[i] >= o[i] else DN
        ax.plot([i, i], [l[i], h[i]], color=col, lw=0.8, zorder=4)
        ax.add_patch(mpatches.Rectangle((i - 0.32, min(o[i], c[i])), 0.64,
                                        max(abs(c[i] - o[i]), a * 0.02),
                                        facecolor=col, edgecolor=col, zorder=5))

    ax.scatter(ph, h[ph], s=24, color="#f97316", zorder=6, label=f"上軌觸點 {len(ph)}")
    ax.scatter(pl, l[pl], s=24, color="#a855f7", zorder=6, label=f"下軌觸點 {len(pl)}")

    ax.axvspan(0, fe, color=LINE, alpha=0.04, zorder=1)
    ax.axvspan(brk, n - 1, color=BREAK, alpha=0.06, zorder=1)
    ax.axvline(brk, color=BREAK, lw=1.5, zorder=2)
    ax.annotate(("突破上軌 " if d > 0 else "跌破下軌 ") + str(df.index[brk].date()),
                (brk, (upper if d > 0 else lower)[brk]),
                xytext=(10, 30 if d > 0 else -40), textcoords="offset points",
                color=BREAK, fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BREAK, lw=1))

    ax.set_xlim(-1, n)
    ax.set_ylim(min(l.min(), lower[n - 1]) * 0.97, max(h.max(), upper[n - 1]) * 1.03)
    fig.suptitle(case["label"], x=0.012, y=0.985, ha="left", color=TXT,
                 fontsize=11.5, fontweight="bold")
    fig.text(0.012, 0.942,
             f"突破／跌破＝該段趨勢終結 → 之後三種可能：上升 / 橫行 / 下跌 ｜ 本例實際走勢：{outcome}",
             color=BREAK, fontsize=9, fontweight="bold")
    ax.set_facecolor(BG)
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.grid(color="#334155", alpha=0.35, lw=0.5)
    for s in ax.spines.values():
        s.set_color("#334155")
    ax.legend(loc="upper left", fontsize=7.5, facecolor="#0f172a", edgecolor="#334155",
              labelcolor=TXT, framealpha=0.85)
    ax.text(0.995, 0.03, case["outcome"], transform=ax.transAxes, color=BREAK,
            fontsize=9.5, fontweight="bold", ha="right")

    axv.bar(x, v, color=[UP if c[i] >= o[i] else DN for i in range(n)], alpha=0.55, width=0.8)
    axv.axvline(brk, color=BREAK, lw=1.2, alpha=0.85)
    axv.set_xlim(-1, n)
    axv.set_facecolor(BG)
    axv.tick_params(colors="#64748b", labelsize=6)
    for s in axv.spines.values():
        s.set_color("#334155")

    fig.text(0.01, 0.006,
             f"資料：yfinance 日線（{df.index[0].date()} → {df.index[-1].date()}）· "
             f"通道以 pivot 回歸擬合（前 {fe} 根棒），突破後淨動 {net:+.1f} ATR",
             color="#475569", fontsize=6.8)
    plt.tight_layout(rect=[0, 0.028, 1, 0.915])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, case["file"])
    fig.savefig(path, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"generated: {path}  (outcome={outcome}, {net:+.1f} ATR)")


if __name__ == "__main__":
    for c in CASES:
        draw(c)
