import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

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
    if vol.iloc[-1] < MIN_VOLUME:
        return False
    avg20 = vol.tail(20).mean()
    avg5 = vol.tail(5).mean()
    avg2 = vol.tail(2).mean()
    return (avg5 < avg20) and (avg2 > avg5)

def is_pullback(df):
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False

    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])

    # 25MAより上
    if close < ma25:
        return False

    # 25MAが上向き
    prev_ma25 = safe_float(df["ma25"].iloc[-6])
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
    """
    IN確率用スコア（0〜100）
    ・RSIゾーン
    ・25MAとの乖離
    ・出来高転換
    ・当日値動き
    を総合評価
    """
    last = df.iloc[-1]
    rsi = safe_float(last.get("rsi"))
    close = safe_float(last.get("close"))
    ma25 = safe_float(last.get("ma25"))
    ret_1d = safe_float(last.get("ret_1d"))

    score = 50.0  # ベース

    # RSI評価
    if np.isfinite(rsi):
        if rsi < 28:
            score += 20   # 強い売られすぎ→反発期待大
        elif rsi < 35:
            score += 15
        elif rsi < 45:
            score += 8
        elif rsi < 60:
            score += 0
        else:
            score -= 10   # 過熱気味

    # 25MAとの距離
    if np.isfinite(close) and np.isfinite(ma25) and ma25 > 0:
        dist = abs(close - ma25) / ma25
        if dist < 0.005:       # ±0.5%以内
            score += 15
        elif dist < 0.01:      # ±1%以内
            score += 10
        elif dist < 0.02:      # ±2%以内
            score += 5
        elif dist > 0.05:      # 5%以上乖離はマイナス
            score -= 10

    # 出来高パターン（is_pullback通過してるので基本OK）
    if volume_ok(df):
        score += 10

    # 当日リターン（ヒゲ・反発の質）
    if np.isfinite(ret_1d):
        ret_pct = ret_1d * 100
        if -3.0 <= ret_pct <= 1.0:
            # 大きく崩れず、小さめの陽線やコマ足
            score += 5
        elif ret_pct > 4.0:
            # 急騰しすぎてると追いかけリスク
            score -= 5

    score = max(0.0, min(100.0, score))
    return int(round(score))

def calc_take_profit(df):
    """
    利確目安（円）
    ・直近10日高値
    ・10MA（ミドル）
    のハイブリッド
    """
    last_close = safe_float(df.iloc[-1]["close"])
    recent_high = safe_float(df["close"].tail(10).max())
    bb_mid = safe_float(df["ma10"].iloc[-1])

    if not np.isfinite(recent_high) or not np.isfinite(bb_mid):
        return int(last_close)

    tp = recent_high * 0.6 + bb_mid * 0.4
    return int(tp)

def calc_stop_loss(df):
    """
    損切りライン（円）
    ・直近5日安値
    ・25MA -1.5%
    ・直近終値 -3%
    のうち最も保守的（低い）ライン
    """
    last_close = safe_float(df.iloc[-1]["close"])
    ma25 = safe_float(df["ma25"].iloc[-1])
    recent_low = safe_float(df["close"].tail(5).min())

    candidates = []

    if np.isfinite(recent_low):
        candidates.append(recent_low)
    if np.isfinite(ma25):
        candidates.append(ma25 * 0.985)
    if np.isfinite(last_close):
        candidates.append(last_close * 0.97)

    if not candidates:
        return int(last_close)

    sl = min(candidates)
    return int(sl)

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
        price = safe_float(last["close"])
        rsi = safe_float(last["rsi"])

        # 買い目安（下限）
        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        candidates = [v for v in [ma5, ma10, price] if np.isfinite(v)]
        if not candidates:
            continue
        lower = int(min(candidates))

        # 理由テキスト
        reasons = []
        if np.isfinite(rsi):
            if rsi <= 32:
                reasons.append("RSIが30前後で売られすぎの強い押し目")
            elif rsi <= 45:
                reasons.append("RSIが中立〜やや売られでちょうど良い押し目")
            else:
                reasons.append("強いトレンド中の浅い押し目")

        ma25 = safe_float(last["ma25"])
        if np.isfinite(ma25) and ma25 > 0:
            dist25 = abs(price - ma25) / ma25
            if dist25 < 0.01:
                reasons.append("25日移動平均線タッチ付近")
            elif dist25 < 0.02:
                reasons.append("25日線近辺での押し目")

        if volume_ok(df):
            reasons.append("出来高が減少から増加に転換（買い需要の出現）")

        in_score = calc_in_score(df)
        tp = calc_take_profit(df)
        sl = calc_stop_loss(df)

        rows.append({
            "ticker": ticker,
            "name": name,
            "price": price,
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

    rank = 1
    for r in cands:
        lower = float(r["lower"])
        tp = float(r["tp"])
        sl = float(r["sl"])
        score = float(r["score"])

        # 勝率目安（ざっくり目安としての参考値）
        win_rate = 30.0 + 0.5 * score   # 30〜80%くらいのレンジ
        win_rate = max(30.0, min(85.0, win_rate))

        # 利確・損切りの％
        if lower > 0:
            tp_pct = (tp / lower - 1.0) * 100.0
            sl_pct = (sl / lower - 1.0) * 100.0
            tp_pct_str = f"{tp_pct:+.1f}%"
            sl_pct_str = f"{sl_pct:+.1f}%"
        else:
            tp_pct_str = "-"
            sl_pct_str = "-"

        lines.append(f"{rank}. {r['ticker']}（{r['name']}）")
        lines.append(f"   IN確率: {int(score)}点（勝率目安: {win_rate:.1f}%）")
        lines.append(f"   買い目安: {int(lower)}円")
        lines.append(f"   利確目安: {int(tp)}円（{tp_pct_str}）")
        lines.append(f"   損切り: {int(sl)}円（{sl_pct_str}）")
        lines.append(f"   理由: {r['reason']}\n")
        rank += 1

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