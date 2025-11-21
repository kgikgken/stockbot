import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

UNIVERSE_CSV_PATH = "universe_jpx.csv"

MIN_HISTORY_DAYS = 30
MIN_VOLUME = 150000
RSI_MIN = 25
RSI_MAX = 70

def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def safe_float(x):
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except:
        return float("nan")

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"])
    df["ticker"] = df["ticker"].astype(str)
    return df

UNIVERSE = load_universe()

def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def fetch_history(ticker):
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
    except:
        return None

    if df is None or df.empty:
        return None

    df = df.tail(60).copy()
    df["close"] = df["Close"].astype(float)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["ret_1d"] = df["close"].pct_change()
    df = add_rsi(df)
    return df

def volume_ok(df):
    vol = df["Volume"].fillna(0)

    if len(vol) < 20:
        return False

    last_vol = float(vol.iloc[-1])
    if last_vol < MIN_VOLUME:
        return False

    avg20 = float(vol.tail(20).mean())
    avg5 = float(vol.tail(5).mean())
    avg2 = float(vol.tail(2).mean())

    cond1 = avg5 < avg20
    cond2 = avg2 > avg5

    return bool(cond1 and cond2)

def is_pullback(df):
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False

    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    prev_ma25 = safe_float(df["ma25"].iloc[-6])

    if not (np.isfinite(close) and np.isfinite(ma25) and np.isfinite(prev_ma25)):
        return False

    # 25MAより上
    if close < ma25:
        return False

    # 25MAが上向き
    if ma25 <= prev_ma25:
        return False

    # RSIゾーン
    rsi = safe_float(last["rsi"])
    if not (RSI_MIN <= rsi <= RSI_MAX):
        return False

    # 出来高パターン
    if not volume_ok(df):
        return False

    return True

def calc_in_score(df):
    last = df.iloc[-1]
    rsi = safe_float(last["rsi"])
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    prev = safe_float(df.iloc[-2]["close"])

    score = 50

    # RSI評価
    if rsi <= 32:
        score += 20
    elif rsi <= 45:
        score += 10

    # 25MAとの近さ
    if np.isfinite(close) and np.isfinite(ma25) and ma25 != 0:
        if abs(close - ma25) / ma25 < 0.01:
            score += 15

    # 出来高
    if volume_ok(df):
        score += 10

    # 当日終値が前日より上なら少し加点
    if np.isfinite(close) and np.isfinite(prev) and close > prev:
        score += 5

    return min(score, 100)

def calc_take_profit(df):
    last = safe_float(df.iloc[-1]["close"])
    recent_high = safe_float(df["close"].tail(10).max())
    bb_mid = safe_float(df["ma10"].iloc[-1])

    if not np.isfinite(recent_high) or not np.isfinite(bb_mid):
        return int(last)

    tp = (recent_high * 0.6 + bb_mid * 0.4)
    return int(tp)

def calc_stop_loss(df):
    last = safe_float(df.iloc[-1]["close"])
    ma25 = safe_float(df["ma25"].iloc[-1])
    recent_low = safe_float(df["close"].tail(5).min())

    loss_candidates = []

    if np.isfinite(recent_low):
        loss_candidates.append(recent_low)
    if np.isfinite(ma25):
        loss_candidates.append(ma25 * 0.985)
    if np.isfinite(last):
        loss_candidates.append(last * 0.97)

    if not loss_candidates:
        return int(last)

    return int(min(loss_candidates))

def pick_top5():
    rows = []

    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        df = fetch_history(ticker)
        if df is None:
            continue

        if not is_pullback(df):
            continue

        last = df.iloc[-1]
        price = safe_float(last["close"])  # 現在株価（直近終値）
        rsi = safe_float(last["rsi"])

        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        lower_candidates = [v for v in [ma5, ma10, price] if np.isfinite(v)]
        if not lower_candidates:
            continue
        lower = int(min(lower_candidates))

        reasons = []
        if np.isfinite(rsi):
            if rsi <= 32:
                reasons.append("売られすぎの強い押し目")
            elif rsi <= 45:
                reasons.append("理想的な押し目")
            else:
                reasons.append("軽めの押し目")

        ma25 = safe_float(last["ma25"])
        if np.isfinite(ma25) and ma25 != 0:
            dist25 = abs(price - ma25) / ma25
            if dist25 < 0.01:
                reasons.append("25MAタッチ")
            elif dist25 < 0.02:
                reasons.append("25MA近辺の押し目")

        if volume_ok(df):
            reasons.append("出来高が減→増へ転換")

        in_score = calc_in_score(df)
        tp = calc_take_profit(df)
        sl = calc_stop_loss(df)

        rows.append({
            "ticker": ticker,
            "name": name,
            "price": price,   # 現在株価
            "lower": lower,
            "rsi": rsi,
            "reason": " / ".join(reasons),
            "score": in_score,
            "tp": tp,
            "sl": sl
        })

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(5)
    return df.to_dict("records")

def build_message():
    now = jst_now().strftime("%Y-%m-%d")
    cands = pick_top5()

    if not cands:
        return f"📉 {now}\n本日の本命TOP5銘柄はありません。"

    lines = []
    lines.append(f"📈 {now} 本日の本命TOP5\n")

    # 詳細パート
    for i, r in enumerate(cands, 1):
        lines.append(f"{i}. {r['ticker']}（{r['name']}）")
        lines.append(f"   IN確率: {r['score']}点")
        lines.append(f"   買い目安: {r['lower']}円（現在 {int(r['price'])}円）")
        lines.append(f"   利確目安: {r['tp']}円")
        lines.append(f"   損切り: {r['sl']}円")
        lines.append(f"   理由: {r['reason']}\n")

    # まとめパート
    lines.append("【まとめ】")
    for r in cands:
        lines.append(f"{r['ticker']}（{r['name']}）: IN確率 {r['score']}点")

    return "\n".join(lines)

def send_line(message):
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("LINE_TOKENが設定されていません")
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {"messages": [{"type": "text", "text": message}]}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE status:", resp.status_code)
        if resp.status_code != 200:
            print("LINE response:", resp.text)
    except Exception as e:
        print("LINE送信エラー:", e)

def main():
    msg = build_message()
    print(msg)  # ログ用
    send_line(msg)

if __name__ == "__main__":
    main()