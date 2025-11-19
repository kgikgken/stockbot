import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# =========================
# 設定まわり
# =========================

UNIVERSE_CSV_PATH = "universe_jpx.csv"  # あれば全銘柄対応になる

# フォールバック用（CSVが無い場合だけ使う簡易ユニバース）
FALLBACK_UNIVERSE: Dict[str, List[tuple]] = {
    "半導体・電子部品": [
        ("8035.T", "東京エレクトロン"),
        ("6920.T", "レーザーテック"),
        ("7751.T", "キヤノン"),
    ],
    "自動車": [
        ("7203.T", "トヨタ"),
        ("7267.T", "ホンダ"),
        ("7201.T", "日産"),
    ],
    "情報通信・インターネット": [
        ("9433.T", "KDDI"),
        ("9432.T", "NTT"),
        ("4755.T", "楽天G"),
    ],
    "商社・資源": [
        ("8058.T", "三菱商事"),
        ("8031.T", "三井物産"),
        ("8001.T", "伊藤忠"),
    ],
}

# ロジック用パラメータ
PULLBACK_MA_TOL = 0.03        # 5/10MAから±3％以内
PULLBACK_LOOKBACK = 3         # 直近3本で押し目判定
PULLBACK_NEG_COUNT = 2        # 3本中2本以上陰線
MIN_HISTORY_DAYS = 30         # 最低データ本数
RSI_MIN = 30.0                # RSI下限（売られすぎ気味〜中立）
RSI_MAX = 65.0                # RSI上限（過熱手前くらい）


# =========================
# 共通ユーティリティ
# =========================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def safe_float(x) -> float:
    """NaN や Series が来ても落ちないように float に変換"""
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float("nan")


# =========================
# ユニバース読み込み
# =========================

def load_universe() -> pd.DataFrame:
    """
    universe_jpx.csv があればそれを使う。
    無ければ FALLBACK_UNIVERSE から DataFrame を作る。
    """
    if os.path.exists(UNIVERSE_CSV_PATH):
        df = pd.read_csv(UNIVERSE_CSV_PATH)
        required_cols = {"ticker", "name", "sector"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"universe_jpx.csv には {required_cols} が必要です")
        df = df.dropna(subset=["ticker", "sector", "name"]).copy()
        df["ticker"] = df["ticker"].astype(str)
        return df

    # フォールバック
    rows = []
    for sector, lst in FALLBACK_UNIVERSE.items():
        for ticker, name in lst:
            rows.append(
                {"ticker": ticker, "name": name, "sector": sector, "market": "Fallback"}
            )
    return pd.DataFrame(rows)


UNIVERSE_DF = load_universe()
TICKER_NAME: Dict[str, str] = dict(
    zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["name"])
)
TICKER_SECTOR: Dict[str, str] = dict(
    zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["sector"])
)


# =========================
# データ取得＆加工
# =========================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSIを計算して rsi 列に入れる"""
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
    yfinance から過去データ取得（最大60営業日分）。
    失敗時は None を返す。
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] {ticker} ダウンロード失敗: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] {ticker} データなし")
        return None

    df = df.tail(60).copy()

    if "Close" not in df.columns:
        print(f"[WARN] {ticker} Close列が存在しません")
        return None

    df["close"] = df["Close"].astype(float)
    df["ret_1d"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()

    df = add_rsi(df)  # RSI追加
    return df


# =========================
# セクター強度計算
# =========================

def calc_sector_strength() -> pd.DataFrame:
    records = []
    grouped = UNIVERSE_DF.groupby("sector")

    for sector, grp in grouped:
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
                ma25_prev_raw = df["ma25"].iloc[-6]
            else:
                base_close = safe_float(df["close"].iloc[0])
                ma25_prev_raw = df["ma25"].iloc[0]

            if base_close <= 0:
                continue

            ret_1d = safe_float(last["ret_1d"]) * 100
            ret_5d = (close_now / base_close - 1) * 100

            ma25_now = safe_float(last["ma25"])
            ma25_prev = safe_float(ma25_prev_raw)
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


# =========================
# 押し目判定ロジック
# =========================

def volume_pattern_ok(df: pd.DataFrame) -> bool:
    """出来高 減少→増加パターンをざっくり判定"""
    if "Volume" not in df.columns:
        return True  # 出来高情報が無い場合は通す

    vols = df["Volume"].tail(15).dropna()
    if len(vols) < 7:
        return True  # データ少ないときは厳しく見ない

    # 前半 vs 直近5本
    recent5 = vols.tail(6).iloc[:-1]   # 今日の1本前まで5本
    prev = vols.iloc[:-6]
    if len(prev) == 0:
        return True

    prev_mean = prev.mean()
    recent_mean = recent5.mean()
    today_vol = vols.iloc[-1]

    # 「直近で出来高が細って → 今日少し増えてきた」イメージ
    if not (recent_mean < prev_mean and today_vol > recent_mean):
        return False

    return True


def is_pullback(df: pd.DataFrame) -> bool:
    """
    押し目判定ロジック（拡張版）

    条件：
      1. 25日線の上にいる
      2. 25日線が5営業日前より上（上向き）
      3. 5日 or 10日線付近（±3％以内）
      4. 直近3本のローソク足で、陰線が多い or もみ合い
      5. RSI が 30〜65 の範囲
      6. 出来高 減少→増加 パターン
    """
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False

    last = df.iloc[-1]

    close_now = safe_float(last["close"])
    ma5_now = safe_float(last["ma5"])
    ma10_now = safe_float(last["ma10"])
    ma25_now = safe_float(last["ma25"])
    rsi_now = safe_float(last.get("rsi", np.nan))

    if not np.isfinite(close_now) or not np.isfinite(ma25_now):
        return False

    # 1. 25日線の上
    if close_now < ma25_now:
        return False

    # 2. 25日線が上向き
    if len(df) < 30:
        return False

    ma25_prev = safe_float(df["ma25"].iloc[-6])
    if not np.isfinite(ma25_prev) or ma25_now <= ma25_prev:
        return False

    # 3. 5日 or 10日線付近（±3％以内）
    cond_ma5 = np.isfinite(ma5_now) and abs(close_now - ma5_now) / ma5_now <= PULLBACK_MA_TOL
    cond_ma10 = np.isfinite(ma10_now) and abs(close_now - ma10_now) / ma10_now <= PULLBACK_MA_TOL
    if not (cond_ma5 or cond_ma10):
        return False

    # 4. 直近3本の終値リターン
    recent = df["ret_1d"].tail(PULLBACK_LOOKBACK).dropna()
    if len(recent) < 2:
        return False

    negatives = (recent < 0).sum()
    last_ret = float(recent.iloc[-1])
    if not (negatives >= PULLBACK_NEG_COUNT or (negatives >= 1 and abs(last_ret) < 0.01)):
        return False

    # 5. RSI フィルター
    if not np.isfinite(rsi_now) or not (RSI_MIN <= rsi_now <= RSI_MAX):
        return False

    # 6. 出来高パターン
    if not volume_pattern_ok(df):
        return False

    return True


# =========================
# 候補銘柄の抽出
# =========================

def pick_candidates(strong_sectors: List[str], per_sector: int = 3) -> pd.DataFrame:
    """
    強いセクターの中から押し目候補を抽出。
    """
    rows = []

    # strong_sectors に含まれる銘柄だけ対象
    target_df = UNIVERSE_DF[UNIVERSE_DF["sector"].isin(strong_sectors)]

    for _, row in target_df.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(ticker)
        if df is None or len(df) < MIN_HISTORY_DAYS:
            continue

        if not is_pullback(df):
            continue

        last = df.iloc[-1]
        price = safe_float(last["close"])
        chg_1d = safe_float(last["ret_1d"]) * 100
        rsi_now = safe_float(last.get("rsi", np.nan))
        ma5_now = safe_float(last["ma5"])
        ma10_now = safe_float(last["ma10"])

        if not np.isfinite(price) or not np.isfinite(chg_1d):
            continue

        # 買い目安レンジ（5日線〜10日線＋終値を含めたレンジ）
        ma_candidates = [v for v in [ma5_now, ma10_now, price] if np.isfinite(v)]
        if ma_candidates:
            buy_lower = min(ma_candidates)
            buy_upper = max(ma_candidates)
        else:
            buy_lower = buy_upper = price

        rows.append(
            {
                "sector": sector,
                "ticker": ticker,
                "name": name,
                "price": price,
                "chg_1d": chg_1d,
                "rsi": rsi_now,
                "buy_lower": buy_lower,
                "buy_upper": buy_upper,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 押しているもの優先（1日リターンの小さい順）
    df = df.sort_values(["sector", "chg_1d"])

    # セクターごとに最大 per_sector 銘柄に絞る
    out = []
    for sector, grp in df.groupby("sector"):
        out.append(grp.head(per_sector))

    return pd.concat(out)


# =========================
# メッセージ生成
# =========================

def build_message() -> str:
    """LINEで送るテキストを組み立てる"""
    # 1. セクター強度
    try:
        sec_df = calc_sector_strength()
    except Exception as e:
        print("[ERROR] calc_sector_strength failed:", e)
        return f"スクリーニング中にエラーが発生しました: {e}"

    if sec_df.empty:
        return "セクター情報が取得できませんでした。"

    # 5日騰落率の強い順に並べてTOP3
    sec_df = sec_df.sort_values("avg_5d", ascending=False)
    top = sec_df.head(3).reset_index(drop=True)
    strong_sectors = list(top["sector"])

    now = jst_now()
    lines: List[str] = []

    # タイトル
    lines.append(f"📈 {now:%Y-%m-%d} スイング候補レポート")
    lines.append("")

    # テーマ候補（セクター）
    lines.append("【今日のテーマ候補（セクターベース）】")
    for _, r in top.iterrows():
        comment = ""
        if r["avg_5d"] > 0 and r["avg_slope25"] > 0:
            comment = "（上昇トレンド継続気味）"
        elif r["avg_5d"] > 0 and r["avg_slope25"] <= 0:
            comment = "（短期強めだがトレンドはフラット気味）"
        elif r["avg_5d"] <= 0 and r["avg_slope25"] > 0:
            comment = "（調整中の押し目候補セクター）"
        else:
            comment = "（全体弱めの中ではマシな部類）"

        lines.append(
            f"- {r['sector']}: 1日 {r['avg_1d']:.1f}% / 5日 {r['avg_5d']:.1f}% / "
            f"25日線傾き {r['avg_slope25']:.2f}% {comment}"
        )

    lines.append("")
    lines.append("【押し目スイング候補】")

    # 2. 押し目候補
    try:
        cands = pick_candidates(strong_sectors)
    except Exception as e:
        print("[ERROR] pick_candidates failed:", e)
        cands = None

    if cands is None or cands.empty:
        lines.append("条件に合う銘柄が見つかりませんでした。")
    else:
        for sector, grp in cands.groupby("sector"):
            lines.append(f"▼{sector}")
            for _, r in grp.iterrows():
                name = r.get("name", "")
                code_name = f"{r['ticker']}（{name}）" if name else r["ticker"]
                buy_lower = r.get("buy_lower", np.nan)
                buy_upper = r.get("buy_upper", np.nan)
                rsi_now = r.get("rsi", np.nan)

                lines.append(
                    f"  - {code_name}: 終値 {r['price']:.1f} 円 / 日中変化 {r['chg_1d']:.1f}%"
                    + (f" / RSI {rsi_now:.1f}" if np.isfinite(rsi_now) else "")
                )
                if np.isfinite(buy_lower) and np.isfinite(buy_upper):
                    lines.append(
                        f"      買うなら: {buy_lower:.0f}〜{buy_upper:.0f} 円目安（5〜10日線付近）"
                    )

    return "\n".join(lines)


# =========================
# LINE 送信まわり
# =========================

def send_line(message: str) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("[ERROR] LINE_TOKEN が設定されていません。")
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {
        "messages": [{"type": "text", "text": message}]
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE API status:", resp.status_code)
        if resp.status_code != 200:
            print("LINE API response body:", resp.text)
    except Exception as e:
        print("[ERROR] LINE送信中にエラー:", e)


# =========================
# エントリポイント
# =========================

def main() -> None:
    msg = build_message()
    send_line(msg)


if __name__ == "__main__":
    main()
