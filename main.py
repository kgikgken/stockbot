import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# 🔧 スクリーニング対象の日本株（簡易ユニバース）
# 必要に応じて銘柄を増やしていけるようにしてある
UNIVERSE = {
    "半導体・電子部品": [
        "8035.T",  # 東京エレクトロン
        "6920.T",  # レーザーテック
        "7751.T",  # キヤノン
    ],
    "自動車": [
        "7203.T",  # トヨタ
        "7267.T",  # ホンダ
        "7201.T",  # 日産
    ],
    "情報通信・インターネット": [
        "9433.T",  # KDDI
        "9432.T",  # NTT
        "4755.T",  # 楽天G
    ],
    "商社・資源": [
        "8058.T",  # 三菱商事
        "8031.T",  # 三井物産
        "8001.T",  # 伊藤忠
    ],
}


def fetch_history(ticker: str, period: str = "3mo") -> pd.DataFrame | None:
    """yfinance から過去データ取得（60営業日分に絞る）"""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.tail(60).copy()
    df["close"] = df["Close"]
    df["ret_1d"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    return df


def calc_sector_strength() -> pd.DataFrame:
    """セクターごとの1日・5日騰落率と25日線の傾きを計算"""
    records = []
    for sector, tickers in UNIVERSE.items():
        vals = []
        for t in tickers:
            df = fetch_history(t)
            if df is None or len(df) < 25:
                continue
            last = df.iloc[-1]
            # 5営業日前との比較（最低6本は欲しい）
            if len(df) >= 6:
                base = df.iloc[-6]
            else:
                base = df.iloc[0]
            ret_5d = (last["close"] / base["close"] - 1) * 100
            ret_1d = last["ret_1d"] * 100

            ma25_now = last["ma25"]
            ma25_prev = df["ma25"].iloc[-6] if len(df) >= 6 else np.nan
            if pd.notna(ma25_now) and pd.notna(ma25_prev) and ma25_prev != 0:
                slope25 = (ma25_now - ma25_prev) / ma25_prev * 100
            else:
                slope25 = 0.0

            vals.append((ret_1d, ret_5d, slope25))

        if not vals:
            continue

        arr = np.array(vals)
        records.append(
            {
                "sector": sector,
                "avg_1d": float(arr[:, 0].mean()),
                "avg_5d": float(arr[:, 1].mean()),
                "avg_slope25": float(arr[:, 2].mean()),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def is_pullback(df: pd.DataFrame) -> bool:
    """押し目判定ロジック（シンプル版）"""
    last = df.iloc[-1]

    # 25日線の上にいるか
    if pd.isna(last["ma25"]) or last["close"] < last["ma25"]:
        return False

    # 25日線が上向き（5営業日前より上）
    if len(df) < 30 or pd.isna(df["ma25"].iloc[-6]):
        return False
    if last["ma25"] <= df["ma25"].iloc[-6]:
        return False

    # 5日 or 10日線付近（±3％以内）
    cond_ma5 = pd.notna(last["ma5"]) and abs(last["close"] - last["ma5"]) / last["ma5"] <= 0.03
    cond_ma10 = pd.notna(last["ma10"]) and abs(last["close"] - last["ma10"]) / last["ma10"] <= 0.03
    if not (cond_ma5 or cond_ma10):
        return False

    # 直近3本のローソク足（終値ベース）の動き：2本以上陰線など
    recent = df["ret_1d"].tail(3)
    negatives = (recent < 0).sum()
    last_ret = recent.iloc[-1]
    if not (negatives >= 2 or (negatives >= 1 and abs(last_ret) < 0.01)):
        return False

    return True


def pick_candidates(strong_sectors: list[str], per_sector: int = 3) -> pd.DataFrame:
    """強いセクターの中から押し目候補を抽出"""
    rows = []
    for sector in strong_sectors:
        for ticker in UNIVERSE.get(sector, []):
            df = fetch_history(ticker)
            if df is None or len(df) < 25:
                continue
            if not is_pullback(df):
                continue
            last = df.iloc[-1]
            rows.append(
                {
                    "sector": sector,
                    "ticker": ticker,
                    "price": float(last["close"]),
                    "chg_1d": float(last["ret_1d"] * 100),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 押しているもの優先でソート（1日リターンが小さい順）
    df = df.sort_values(["sector", "chg_1d"])

    # セクターごとに最大 per_sector 銘柄に絞る
    out = []
    for sector, grp in df.groupby("sector"):
        out.append(grp.head(per_sector))
    return pd.concat(out)


def build_message() -> str:
    """LINEで送るテキストを組み立てる"""
    try:
        sec_df = calc_sector_strength()
    except Exception as e:
        return f"スクリーニング中にエラーが発生しました: {e}"

    if sec_df.empty:
        return "セクター情報が取得できませんでした。"

    # 5日騰落率の強い順に並べてTOP3
    sec_df = sec_df.sort_values("avg_5d", ascending=False)
    top = sec_df.head(3)
    strong_sectors = list(top["sector"])

    # 押し目候補抽出
    try:
        cands = pick_candidates(strong_sectors)
    except Exception as e:
        cands = None

    jst = datetime.now(timezone(timedelta(hours=9)))
    lines: list[str] = []
    lines.append(f"📈 {jst:%Y-%m-%d} スイング候補レポート")
    lines.append("")
    lines.append("【強いセクター TOP3（5日騰落率ベース）】")
    for _, r in top.iterrows():
        lines.append(
            f"- {r['sector']}: 1日 {r['avg_1d']:.1f}% / 5日 {r['avg_5d']:.1f}% / 25日線傾き {r['avg_slope25']:.2f}%"
        )

    lines.append("")
    lines.append("【押し目スイング候補】")
    if cands is None or cands.empty:
        lines.append("条件に合う銘柄が見つかりませんでした。")
    else:
        for sector, grp in cands.groupby("sector"):
            lines.append(f"▼{sector}")
            for _, r in grp.iterrows():
                lines.append(
                    f"  - {r['ticker']}: 終値 {r['price']:.1f} 円 / 日中変化 {r['chg_1d']:.1f}%"
                )

    return "\n".join(lines)


def send_line(message: str) -> None:
    """LINE にテキストメッセージを送る"""
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("LINE_TOKEN が設定されていません。")
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {
        "messages": [
            {"type": "text", "text": message}
        ]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE API status:", resp.status_code, resp.text)
    except Exception as e:
        print("LINE送信中にエラー:", e)


def main():
    msg = build_message()
    send_line(msg)


if __name__ == "__main__":
    main()
