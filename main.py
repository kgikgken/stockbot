import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

# =============================
# 基本設定
# =============================

UNIVERSE_CSV_PATH = "universe_jpx.csv"

# 押し目ロジックの設定
PULLBACK_MA_TOL = 0.05        # MA乖離 ±5%
MIN_HISTORY_DAYS = 30
RSI_MIN = 25.0
RSI_MAX = 75.0

# セクター抽出数
TOP_SECTOR_COUNT = 5

# ACDEスコアの重み
WEIGHT_RSI = 0.5
WEIGHT_MA25 = 0.3
WEIGHT_VOLUME = 0.2


# =============================
# ユーティリティ
# =============================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def safe_float(x) -> float:
    """Series や NaN が来ても float に変換"""
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float("nan")
# =============================
# ユニバース読み込み
# =============================

def load_universe() -> pd.DataFrame:
    """universe_jpx.csv を読み込み、整形して返す"""
    if not os.path.exists(UNIVERSE_CSV_PATH):
        raise FileNotFoundError(f"{UNIVERSE_CSV_PATH} が見つかりません")

    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"]).copy()

    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)
    df["industry_big"] = df["industry_big"].astype(str)
    df["market"] = df["market"].astype(str)

    return df


UNIVERSE_DF = load_universe()
TICKER_NAME: Dict[str, str] = dict(zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["name"]))
TICKER_SECTOR: Dict[str, str] = dict(zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["sector"]))


# =============================
# データ取得 & 加工
# =============================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSIを計算して rsi 列に追加"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def fetch_history(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    yfinance から60営業日分を取得。
    成功時は DataFrame、失敗時は None。
    """
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False
        )
    except Exception:
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    df = df.tail(60).copy()
    df["close"] = df["Close"].astype(float)
    df["ret_1d"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()

    df = add_rsi(df)

    return df
# =============================
# セクター強度計算
# =============================

def calc_sector_strength() -> pd.DataFrame:
    """各セクターの1日・5日騰落率と25日線傾きを計算"""
    records = []

    for sector, grp in UNIVERSE_DF.groupby("sector"):
        vals = []

        for _, row in grp.iterrows():
            ticker = row["ticker"]
            df = fetch_history(ticker)
            if df is None or len(df) < 25:
                continue

            last = df.iloc[-1]
            close_now = safe_float(last["close"])

            if len(df) >= 6:
                base_close = safe_float(df["close"].iloc[-6])
                ma25_prev = safe_float(df["ma25"].iloc[-6])
            else:
                base_close = safe_float(df["close"].iloc[0])
                ma25_prev = safe_float(df["ma25"].iloc[0])

            if base_close <= 0:
                continue

            ret_1d = safe_float(last["ret_1d"]) * 100
            ret_5d = (close_now / base_close - 1) * 100

            ma25_now = safe_float(last["ma25"])
            if np.isfinite(ma25_now) and np.isfinite(ma25_prev) and ma25_prev != 0:
                slope25 = (ma25_now - ma25_prev) / ma25_prev * 100
            else:
                slope25 = 0.0

            if not np.isfinite(ret_1d) or not np.isfinite(ret_5d):
                continue

            vals.append((ret_1d, ret_5d, slope25))

        if not vals:
            continue

        arr = np.array(vals, dtype=float)
        records.append({
            "sector": sector,
            "avg_1d": float(arr[:, 0].mean()),
            "avg_5d": float(arr[:, 1].mean()),
            "avg_slope25": float(arr[:, 2].mean())
        })

    return pd.DataFrame(records)


# =============================
# 出来高パターン判定
# =============================

def volume_pattern_ok(df: pd.DataFrame) -> bool:
    """
    出来高の「減少 → 増加」パターンを判定
    ・直近5日間の出来高平均が、過去20日平均より低い → 減少局面
    ・直近2日間の出来高平均が、直近5日平均より高い → 増加転換
    """
    if "Volume" not in df.columns:
        return False

    vol = df["Volume"].fillna(0)

    if len(vol) < 20:
        return False

    # 過去20日平均
    avg20 = vol.tail(20).mean()

    # 直近5日平均
    avg5 = vol.tail(5).mean()

    # 直近2日平均
    avg2 = vol.tail(2).mean()

    # 減少 → 増加転換を判定
    cond_decrease = avg5 < avg20
    cond_increase = avg2 > avg5

    return bool(cond_decrease and cond_increase)



# =============================
# 押し目判定ロジック
# =============================

def is_pullback(df: pd.DataFrame) -> bool:
    """押し目判定ロジック（RSI / MA / 25MA / 出来高など）"""
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False

    last = df.iloc[-1]
    close_now = safe_float(last["close"])
    ma5 = safe_float(last["ma5"])
    ma10 = safe_float(last["ma10"])
    ma25 = safe_float(last["ma25"])

    if not np.isfinite(close_now) or not np.isfinite(ma25):
        return False

    # 1. 25日線の上
    if close_now < ma25:
        return False

    # 2. 25日線が上向き
    if len(df) < 30:
        return False
    ma25_prev = safe_float(df["ma25"].iloc[-6])
    if not np.isfinite(ma25_prev) or ma25 <= ma25_prev:
        return False

    # 3. MA乖離 ±5%
    cond_ma5 = np.isfinite(ma5) and abs(close_now - ma5) / ma5 <= PULLBACK_MA_TOL
    cond_ma10 = np.isfinite(ma10) and abs(close_now - ma10) / ma10 <= PULLBACK_MA_TOL
    if not (cond_ma5 or cond_ma10):
        return False

    # 4. RSI
    rsi = safe_float(last.get("rsi", np.nan))
    if not (RSI_MIN <= rsi <= RSI_MAX):
        return False

    # 5. 出来高パターン
    if not volume_pattern_ok(df):
        return False

    return True


# =============================
# TOP5 セクター内の押し目銘柄
# =============================

def pick_candidates_in_sector(strong_sectors: List[str]) -> pd.DataFrame:
    """TOP5セクター内の押し目銘柄を抽出"""
    rows = []

    target_df = UNIVERSE_DF[UNIVERSE_DF["sector"].isin(strong_sectors)]

    for _, row in target_df.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(ticker)
        if df is None or not is_pullback(df):
            continue

        last = df.iloc[-1]
        price = safe_float(last["close"])
        chg_1d = safe_float(last["ret_1d"]) * 100
        rsi = safe_float(last.get("rsi"))

        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        buy_lower = min([v for v in [ma5, ma10, price] if np.isfinite(v)])
        buy_upper = max([v for v in [ma5, ma10, price] if np.isfinite(v)])

        rows.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": price,
            "chg_1d": chg_1d,
            "rsi": rsi,
            "buy_lower": buy_lower,
            "buy_upper": buy_upper
        })

    return pd.DataFrame(rows)
# =============================
# セクター外の押し目候補（ACDE複合スコア）
# =============================

def pick_candidates_outside_sector(strong_sectors: List[str]) -> pd.DataFrame:
    """TOP5以外のセクターから押し目銘柄を抽出し、ACDE複合スコアで評価"""
    rows = []

    outside_df = UNIVERSE_DF[~UNIVERSE_DF["sector"].isin(strong_sectors)]

    for _, row in outside_df.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(ticker)
        if df is None or not is_pullback(df):
            continue

        last = df.iloc[-1]

        price = safe_float(last["close"])
        chg_1d = safe_float(last["ret_1d"]) * 100
        rsi = safe_float(last.get("rsi"))

        # 5MA・10MAから買いレンジ
        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        buy_lower = min([v for v in [ma5, ma10, price] if np.isfinite(v)])
        buy_upper = max([v for v in [ma5, ma10, price] if np.isfinite(v)])

        # MA25乖離
        ma25 = safe_float(last["ma25"])
        if ma25 > 0:
            ma25_dis = abs(price - ma25) / ma25
        else:
            ma25_dis = np.nan

        # 出来高転換スコア = 今日出来高 / 最近5日平均
        vol = df["Volume"].dropna()
        if len(vol) >= 6:
            recent5 = vol.tail(6).iloc[:-1]
            vol_score = safe_float(vol.iloc[-1] / recent5.mean())
        else:
            vol_score = 1.0

        # ACDEスコア（低い順が強い押し目）
        score = (
            (rsi if np.isfinite(rsi) else 100) * WEIGHT_RSI +
            (ma25_dis if np.isfinite(ma25_dis) else 1.0) * WEIGHT_MA25 +
            (-vol_score) * WEIGHT_VOLUME  # 出来高増加は良いのでマイナス
        )

        rows.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": price,
            "chg_1d": chg_1d,
            "rsi": rsi,
            "buy_lower": buy_lower,
            "buy_upper": buy_upper,
            "score": score
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("score")  # スコアが低いほど良い押し目

    return df


# =============================
# メッセージ生成
# =============================

def build_message() -> str:
    """LINEで送るメッセージ本文を作成"""

    # --- セクター強度 ---
    sec_df = calc_sector_strength()
    if sec_df.empty:
        return "セクター情報が取得できませんでした。"

    sec_df = sec_df.sort_values("avg_5d", ascending=False)
    top = sec_df.head(TOP_SECTOR_COUNT).reset_index(drop=True)
    strong_sectors = list(top["sector"])

    now = jst_now()
    lines = []
    lines.append(f"📈 {now:%Y-%m-%d} スイング候補レポート\n")

    # --- TOP5セクター ---
    lines.append("【今日のテーマ候補（セクターベース）】")
    for _, r in top.iterrows():
        comment = ""
        if r["avg_5d"] > 0 and r["avg_slope25"] > 0:
            comment = "（強い上昇トレンド）"
        elif r["avg_5d"] > 0:
            comment = "（短期強め）"
        elif r["avg_slope25"] > 0:
            comment = "（押し目セクター）"
        else:
            comment = "（相対的にマシ）"

        lines.append(
            f"- {r['sector']}: 1日 {r['avg_1d']:.1f}% / "
            f"5日 {r['avg_5d']:.1f}% / 25日線傾き {r['avg_slope25']:.2f}% "
            f"{comment}"
        )

    # --- TOP5内銘柄 ---
    cands_in = pick_candidates_in_sector(strong_sectors)

    lines.append("\n【押し目スイング候補（TOP5セクター内）】")
    if cands_in.empty:
        lines.append("条件に合う銘柄がありません。")
    else:
        for sector, grp in cands_in.groupby("sector"):
            lines.append(f"▼{sector}")
            for _, r in grp.iterrows():
                lines.append(
                    f"  - {r['ticker']}（{r['name']}）: 終値 {r['price']:.1f}円 / "
                    f"日中変化 {r['chg_1d']:.1f}% / RSI {r['rsi']:.1f}"
                )
                lines.append(
                    f"      買うなら: {r['buy_lower']:.0f}〜{r['buy_upper']:.0f} 円"
                )

    # --- セクター外候補（ACDE複合スコア） ---
    cands_out = pick_candidates_outside_sector(strong_sectors)

    lines.append("\n【セクター外おすすめ押し目銘柄】")
    if cands_out.empty:
        lines.append("セクター外では押し目優良銘柄なし。")
    else:
        for sector, grp in cands_out.groupby("sector"):
            lines.append(f"▼{sector}")
            for _, r in grp.iterrows():
                lines.append(
                    f"  - {r['ticker']}（{r['name']}）: 終値 {r['price']:.1f}円 / "
                    f"日中変化 {r['chg_1d']:.1f}% / RSI {r['rsi']:.1f}"
                )
                lines.append(
                    f"      買うなら: {r['buy_lower']:.0f}〜{r['buy_upper']:.0f} 円"
                )

    return "\n".join(lines)


# =============================
# LINE送信
# =============================

def send_line(message: str) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("[ERROR] LINE_TOKEN がありません")
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {"messages": [{"type": "text", "text": message}]}

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE API:", resp.status_code)
        if resp.status_code != 200:
            print("Response:", resp.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)


# =============================
# main()
# =============================

def main():
    msg = build_message()
    send_line(msg)


if __name__ == "__main__":
    main()
