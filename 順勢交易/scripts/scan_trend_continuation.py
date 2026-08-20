"""
順勢交易 Trend Continuation 掃描器
框架：前推進浪頂（前頂）突破 → 回測前頂支持位 → 準備延續
條件（對應筆記）：
1. 趨勢成立：近 6 個月有明確上升結構（低點抬升 + 高點抬升）
2. 突破前頂：價格曾明顯升穿前推進浪頂（Breakout，幅度 5-30%，不能太短/太長）
3. 回測中：現價回到前頂附近（前頂 ±4%），未跌穿（支持位之上）
4. 假突破底部止損：回測低點 < 前頂，但現價 > 回測低點（急速返回）
5. 動能：突破時放量（成交量 > 20 日均量 1.5x）
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sys, json

WATCHLIST = [
    # 大型科技
    'NVDA', 'AMD', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'TSLA', 'NFLX',
    # 半導體
    'AVGO', 'MU', 'TXN', 'INTC', 'QCOM', 'ASML', 'AMAT', 'LRCX', 'KLAC',
    # AI / 雲
    'PLTR', 'CRM', 'ORCL', 'SNOW', 'DDOG', 'NET', 'CRWD', 'PANW', 'NOW',
    # 金融
    'JPM', 'GS', 'BAC', 'MS', 'V', 'MA', 'AXP', 'COIN',
    # 醫療 / 生物科技
    'MRNA', 'PFE', 'LLY', 'UNH', 'JNJ', 'ABBV', 'AMGN', 'GILD', 'REGN',
    # 消費
    'WMT', 'COST', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW',
    # 工業 / 能源
    'CAT', 'DE', 'BA', 'GE', 'RTX', 'XOM', 'CVX', 'COP',
    # 其他熱門
    'TSM', 'UBER', 'SHOP', 'SQ', 'SOFI', 'RIVN', 'MSTR', 'COIN', 'HOOD', 'MARA',
]

def find_swing_highs(df, window=5):
    highs = []
    for i in range(window, len(df)-window):
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            highs.append((i, df['High'].iloc[i]))
    return highs

def find_swing_lows(df, window=5):
    lows = []
    for i in range(window, len(df)-window):
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            lows.append((i, df['Low'].iloc[i]))
    return lows

def scan(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period='9mo', interval='1d')
        if df.empty or len(df) < 80:
            return None
        df = df[['Open','High','Low','Close','Volume']].dropna()
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']
        cur = close.iloc[-1]

        # 0. THREE PUSHES 檢查（最優先！）：最近 3 個 swing high 依次升高
        #    且現價由第 3 頂回落 >3% → 力竭反轉，直接排除並標記
        def find_swing_highs_in(df_, window=4):
            out = []
            for j in range(window, len(df_)-window):
                if df_['High'].iloc[j] == df_['High'].iloc[j-window:j+window].max():
                    out.append((j, df_['High'].iloc[j]))
            return out
        sh = find_swing_highs_in(df)
        sh_recent = [x for x in sh if x[0] < len(df)-3][-3:]
        if len(sh_recent) == 3:
            _, h1 = sh_recent[0]; _, h2 = sh_recent[1]; _, h3 = sh_recent[2]
            if h2 > h1 * 1.02 and h3 > h2 * 1.01:
                drop = (cur - h3) / h3 * 100
                if drop < -3:
                    return {'ticker': ticker, 'skip': 'three_pushes',
                            'reason': f'3頂依次升 {h1:.1f}→{h2:.1f}→{h3:.1f}，現價距第3頂 {drop:.1f}% ← 力竭反轉'}

        # 1. 上升結構：近 60 日低點 > 90 日前低點（低點抬升）
        recent_low = low.iloc[-60:].min()
        prior_low = low.iloc[-120:-60].min()
        if recent_low <= prior_low * 1.02:
            return None  # 無明顯上升結構

        # 2. 前推進浪頂：近 120 日 swing high（非最近 10 日，因為可能已突破）
        highs = find_swing_highs(df, 5)
        candidates = [h for h in highs if h[0] < len(df) - 10]
        if not candidates:
            return None
        # 揀最高嗰個前頂（最有代表性）
        prev_peak_idx, prev_peak = max(candidates, key=lambda x: x[1])

        # 3. 突破：近期（最近 30 日內）曾明顯升穿前頂 5-30%
        recent_high = high.iloc[-30:].max()
        if recent_high < prev_peak * 1.05:
            return None  # 未突破
        if recent_high > prev_peak * 1.40:
            return None  # 突破太長（趨勢可能終結，筆記注意重點）

        # 4. 回測中：現價回到前頂附近（前頂 -8% 到 +15%），未跌穿
        if cur < prev_peak * 0.92:
            return None  # 已跌穿前頂（支持失敗）
        if cur > prev_peak * 1.15:
            return None  # 已離前頂太遠（已爆升，追唔到）

        # 5. 回測低點 < 前頂（假突破底部），現價 > 回測低點
        retrace_low = low.iloc[-10:].min()
        if retrace_low >= prev_peak:
            return None  # 未回測到前頂（回調唔夠深）
        if cur <= retrace_low:
            return None  # 仍喺低位（未反彈）

        # 6. 突破放量
        break_idx = np.where(high.iloc[-30:] >= prev_peak)[0]
        if len(break_idx) == 0:
            return None
        first_break = len(df) - 30 + int(break_idx[0])
        vol_break = vol.iloc[max(0,first_break-2):first_break+2].mean()
        vol_avg = vol.iloc[-120:].mean()
        if vol_break < vol_avg * 1.2:
            return None  # 突破冇放量

        # 回報
        return {
            'ticker': ticker,
            'price': round(cur, 2),
            'prev_peak': round(prev_peak, 2),
            'retrace_low': round(retrace_low, 2),
            'retrace_pct': round((cur - prev_peak) / prev_peak * 100, 1),
            'breakout_high': round(recent_high, 2),
            'breakout_pct': round((recent_high - prev_peak) / prev_peak * 100, 1),
            'vol_ratio': round(vol_break / vol_avg, 1),
            'days_ago': len(df) - 1 - prev_peak_idx,
        }
    except Exception as e:
        return None

if __name__ == '__main__':
    results = []
    skipped_tp = []
    failed = []
    for i, tk in enumerate(WATCHLIST):
        r = scan(tk)
        if r:
            if r.get('skip') == 'three_pushes':
                skipped_tp.append(r)
            else:
                results.append(r)
        else:
            failed.append(tk)
        sys.stdout.write(f'\r掃描中 {i+1}/{len(WATCHLIST)}: {tk}  {"FOUND" if r else "..."}    ')
        sys.stdout.flush()
    print()
    print(f'\n=== 順勢交易候選（突破後回測前頂） ===')
    if not results:
        print('（今次掃描冇發現符合條件嘅股票）')
    for r in sorted(results, key=lambda x: x['retrace_pct']):
        print(f"{r['ticker']:6s} 現價 ${r['price']:8.2f} | 前頂 ${r['prev_peak']:8.2f} | 回測低點 ${r['retrace_low']:8.2f} | 現價距前頂 {r['retrace_pct']:+.1f}% | 突破 +{r['breakout_pct']:.0f}% | 量比 {r['vol_ratio']}x | 前頂喺 {r['days_ago']} 日前")
    if skipped_tp:
        print(f'\n⚠️ THREE PUSHES 排除（力竭反轉，勿當順勢）:')
        for r in skipped_tp:
            print(f"  {r['ticker']:6s} {r['reason']}")
    print(f'\n掃描 {len(WATCHLIST)} 隻，無信號 {len(failed)} 隻，three pushes 排除 {len(skipped_tp)} 隻')
