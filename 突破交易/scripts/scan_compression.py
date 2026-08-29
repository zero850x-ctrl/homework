#!/usr/bin/env python3
"""992-style pre-breakout compression scanner for HK or US equities.

Usage:
    python3 scan_compression.py hk   # Hong Kong stocks
    python3 scan_compression.py us   # US stocks (NASDAQ/NYSE only)

Pipeline:
    1. TradingView scanner API pre-filter: near 52-week high + liquidity.
    2. yfinance FULL-HISTORY OHLCV -> 6-point compression score.
    3. Prints A/B grade candidates (score >= 4).

Key correctness rule (user correction 2026-08-16):
    Proximity is measured to the TRUE all-time high (ATH over full history),
    NOT a 90-day high. A 90d/120d resistance is still used for touch-counting
    and compression math, but "near high" must mean near ATH. Example: MA's
    true ATH is $601.77 while its 90d high was $583.71 — a 90d-based scan
    wrongly ranked it as "near high".

HK data gap warning (user request 2026-08-16):
    Yahoo Finance has NO historical data for some listed HK blue chips
    (e.g. 941 中移動, 178 莎莎, 823 領展, 215 和記電訊). They report
    "possibly delisted" although they ARE trading. The scanner silently
    skips them — so HK results are INCOMPLETE. The script now counts and
    reports skipped tickers so the user knows the result is partial.

Output: score, ticker, company, close, ATH, prox_ath, prox120, touches,
        higher_lows, compression ratio, volume contraction.
"""
import sys
import time
import json

import numpy as np
import requests
import yfinance as yf

try:
    from tvDatafeed import TvDatafeed, Interval as TVInterval
    _TV = TvDatafeed()
    _TV_AVAILABLE = True
except Exception:
    _TV_AVAILABLE = False

TV_URLS = {"hk": "https://scanner.tradingview.com/hongkong/scan",
           "us": "https://scanner.tradingview.com/america/scan"}

TV_COLS = ["name", "description", "close", "change", "volume",
           "price_52_week_high", "price_52_week_low", "market_cap_basic",
           "relative_volume_10d_calc", "RSI", "average_volume_30d_calc", "exchange"]

# Market-specific pre-filters (liquidity / size)
MIN_CAP = {"hk": 5e8, "us": 3e9}          # USD market cap floor
MIN_VOL30 = {"hk": 100_000, "us": 300_000}  # 30d avg volume floor
MIN_PROX = {"hk": 0.90, "us": 0.92}       # close / 52w-high floor for pre-filter


def tv_scan(market: str):
    filters = [{"left": "type", "operation": "equal", "right": "stock"},
               {"left": "close", "operation": "greater", "right": 0},
               {"left": "market_cap_basic", "operation": "greater", "right": MIN_CAP[market]}]
    if market == "us":
        # CRITICAL: america/scan includes OTC/foreign 5-letter listings (MBFJF etc.)
        filters.append({"left": "exchange", "operation": "in_range",
                        "right": ["NASDAQ", "NYSE"]})
    payload = {"symbols": {"query": {"types": []}, "tickers": []},
               "columns": TV_COLS,
               "filter": filters,
               "options": {"lang": "en"},
               "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
               "range": [0, 3000]}
    resp = requests.post(TV_URLS[market], json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    cands = []
    for x in data:
        d = dict(zip(TV_COLS, x["d"]))
        try:
            close = float(d["close"])
            high52 = float(d["price_52_week_high"])
            if close > 0 and high52 > 0 and close >= MIN_PROX[market] * high52:
                if float(d.get("average_volume_30d_calc") or 0) >= MIN_VOL30[market]:
                    cands.append(d)
        except (TypeError, ValueError):
            continue
    return cands


def compression_score(df):
    """6-point compression score vs TRUE ATH. None if insufficient data.

    Points (all must use last 120 bars for compression math):
      1. prox_ath >= 0.90   (close / all-time high over FULL history)
      2. touches >= 2       (high >= 0.985 * 120d-high in last 25 bars)
      3. higher lows        (min low[-10:] > min low[-20:-10])
      4. range compression  (mean range[-5:] / mean range[-15:-5] < 0.85)
      5. volume contraction (mean vol[-5:] / mean vol[-15:-5] < 0.9)
      6. near-high close    (close >= 0.97 * 120d-high)
    """
    if df is None or len(df) < 60:
        return None
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    v = df["Volume"].values
    if np.isnan(h[-1]) or h[-1] <= 0:
        return None
    ATH = float(np.max(h))                        # TRUE all-time high (full history)
    if ATH <= 0:
        return None
    H, L, C, V = h[-120:], l[-120:], c[-120:], v[-120:]
    R = float(np.max(H))                          # 120d high = recent resistance
    prox_ath = C[-1] / ATH                        # proximity to TRUE ATH
    # ── 崩盤股偵測 (2026-08-25): ATH 高現價 3 倍+ = 唔係 992 壓縮,
    # 係崩盤後反彈 (例 8239: 2007 見 $792, 現 $21)。prox gate 會過濾,
    # 但要明確標示原因, 唔好同「數據缺口」混淆。
    crash_flag = ATH > C[-1] * 3
    prox120 = C[-1] / R
    touches = int(np.sum(H[-25:] >= 0.985 * R))   # touches of resistance zone
    hl = bool(L[-10:].min() > L[-20:-10].min())   # higher lows
    def rng(s):
        return float(np.mean(H[s] - L[s]))
    comp = rng(slice(-5, None)) / max(rng(slice(-15, -5)), 1e-9)   # range compression
    vc = float(np.mean(V[-5:])) / max(float(np.mean(V[-15:-5])), 1e-9)  # volume contraction
    nearhi = C[-1] >= 0.97 * R

    # ── Fib 0.236 gate (筆記四大條件②, 2026-08-25 對齊 homework page) ──
    # anchor = 120 日 impulse 起點低位；收盤必須企穩喺 0.236 回撤之上，
    # 跌破 = 信號失效（參考: 中電 0002 97.35 見頂後連破 0.236/0.382/0.618）
    anchor = float(np.min(L))
    fib236 = R - 0.236 * (R - anchor)
    fib_ok = bool(C[-1] > fib236)

    # ── 四大條件 (筆記: 缺一不可) ──
    # ①歷史高位 prox_ath>=0.90 ②EQH touches>=2 ③收窄 comp<0.85 ④量縮 vc<0.9
    big4 = bool(prox_ath >= 0.90 and touches >= 2 and comp < 0.85 and vc < 0.9)

    # ── trade management 參考價 (筆記紀律 02/03) ──
    buildup_low = float(np.min(L[-40:]))          # build-up 最低位
    buildup_h = R - buildup_low
    sl_ref = round(buildup_low, 2)                # 止損: build-up 底下方
    tp1_ref = round(R + 1.5 * buildup_h, 2) if buildup_h > 0 else None   # TP1 = build-up 1.5 倍
    tp2_ref = round(ATH + 0.618 * (ATH - anchor), 2) if ATH > anchor else None  # TP2 = 趨勢 61.8% 延伸

    # ── 失敗模式 flags (筆記失敗例子) ──
    flags = []
    if comp > 1.0:
        flags.append("range_expanding")           # 波幅擴緊 = 冇蓄勢
    if len(C) > 1 and V[-1] > 1.5 * max(float(np.mean(V[-15:-5])), 1e-9) and C[-1] < C[-2]:
        flags.append("red_volume_spike")          # 放量大陰線 = 派發疑雲

    s = 0
    if prox_ath >= 0.90: s += 1
    if touches >= 2: s += 1
    if hl: s += 1
    if comp < 0.85: s += 1
    if vc < 0.9: s += 1
    if nearhi: s += 1
    return {"prox_ath": round(prox_ath, 4), "prox120": round(prox120, 4),
            "touches": touches, "hl": hl, "comp": round(comp, 2),
            "vc": round(vc, 2), "score": s, "ATH": round(ATH, 2),
            "R": round(R, 2), "close": round(float(C[-1]), 2),
            "fib236": round(fib236, 2), "fib_ok": fib_ok, "big4": big4,
            "SL": sl_ref, "TP1": tp1_ref, "TP2": tp2_ref,
            "crash": crash_flag,
            "flags": ",".join(flags) if flags else ""}


def fetch_hist(market: str, ticker: str):
    """Fetch full-history daily OHLCV. HK prefers tvdatafeed (yfinance has gaps).

    Returns a DataFrame with capitalized columns, or None.
    """
    if market == "us":
        df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
        if df is not None and len(df.columns) > 1:
            df.columns = [x[0] for x in df.columns]
        return df
    # HK: tvdatafeed first (no leading zero, e.g. '941' not '0941')
    if _TV_AVAILABLE:
        try:
            df = _TV.get_hist(symbol=ticker, exchange='HKEX', interval=TVInterval.in_daily, n_bars=800)
            if df is not None and len(df) > 0:
                df = df.rename(columns={'open': 'Open', 'high': 'High',
                                        'low': 'Low', 'close': 'Close',
                                        'volume': 'Volume'})
                return df
        except Exception:
            pass
    # fallback: yfinance
    try:
        df = yf.download(f"{ticker}.HK", period="max", interval="1d", progress=False, auto_adjust=False)
        if df is not None and len(df.columns) > 1:
            df.columns = [x[0] for x in df.columns]
        return df
    except Exception:
        return None


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "us"
    if market not in TV_URLS:
        print("usage: scan_compression.py hk|us")
        sys.exit(1)
    cands = tv_scan(market)
    cands.sort(key=lambda d: float(d["close"]) / float(d["price_52_week_high"]), reverse=True)
    print(f"[*] {market.upper()} candidates near 52w high: {len(cands)}", flush=True)
    if market == "hk" and not _TV_AVAILABLE:
        print("[!] tvdatafeed NOT available — HK scan will have data gaps. pip install tvdatafeed", flush=True)

    results = []
    skipped = []        # tickers where no data source works (HK data gap)
    for d in cands:
        ticker = d["name"]
        try:
            df = fetch_hist(market, ticker)
            if df is None or len(df) == 0:
                skipped.append((ticker, d["description"], "no data (gap)"))
                continue
            s = compression_score(df)
            # 崩盤股過濾 (筆記②: 歷史高位 = 強勢動能, 崩盤反彈唔算)
            if s and s.get("crash"):
                skipped.append((ticker, d["description"], "post-crash rebound (ATH > 3x, not 992 pattern)"))
                continue
            # Fib 0.236 gate (筆記: 跌破 0.236 = 信號失效) — 先決條件
            if s and not s["fib_ok"]:
                skipped.append((ticker, d["description"], "below Fib 0.236 (signal invalid)"))
                continue
            # require true-ATH proximity; report score >= 4
            if s and s["prox_ath"] >= 0.85 and s["score"] >= 4:
                results.append((s["score"], ticker, d["description"], s))
        except Exception as e:
            skipped.append((ticker, d["description"], str(e)[:40]))
        time.sleep(0.15)   # rate-limit throttle

    results.sort(key=lambda r: (not r[3].get("big4", False), -r[0]))  # big4 先, 再 by score
    print(f"=== ATH-based score>=4: {len(results)} (A=四大條件齊 ★) ===", flush=True)
    for sc, t, desc, s in results:
        star = "★" if s.get("big4") else " "
        flag = f" ⚠️{s['flags']}" if s.get("flags") else ""
        tp1 = f" TP1={s['TP1']}" if s.get("TP1") else ""
        tp2 = f" TP2={s['TP2']}" if s.get("TP2") else ""
        print(f"{star}{sc} {t:6s} {desc[:30]:30s} close={s['close']:9.2f} ATH={s['ATH']:9.2f} "
              f"proxATH={s['prox_ath']} touches={s['touches']} comp={s['comp']} vc={s['vc']}"
              f" SL={s['SL']}{tp1}{tp2}{flag}", flush=True)

    # data-gap warning (HK: tvdatafeed+yfinance both can miss names)
    if skipped:
        print(f"\n⚠️  {market.upper()} data gap: {len(skipped)}/{len(cands)} candidates skipped — "
              f"no historical data available. Result is INCOMPLETE.", flush=True)
        for t, desc, why in skipped[:15]:
            print(f"   - {t} {desc[:28]:28s} ({why})", flush=True)
        if len(skipped) > 15:
            print(f"   ... and {len(skipped)-15} more", flush=True)


if __name__ == "__main__":
    main()
