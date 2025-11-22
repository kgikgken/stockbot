import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ================================
# 基本設定
# ================================

UNIVERSE_CSV_PATH = "universe_jpx.csv"

MIN_HISTORY_DAYS = 30
MIN_VOLUME = 150000
RSI_MIN = 25
RSI_MAX = 70

# セクターごとのざっくり分類（地合い連動スコア用）
DEFENSIVE_SECTORS = [
    "電気・ガス業", "食料品", "医薬品", "陸運業", "空運業",
    "小売業", "サービス業"
]
RISK_SECTORS = [
    "情報・通信業", "電気機器", "機械", "精密機器", "非鉄金属",
    "金属製品", "証券、商品先物取引業", "その他金融業"
]


# ================================
# ユーティリティ
# ================================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def safe_float(x) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"])
    df["ticker"] = df["ticker"].astype(str)
    df["sector"] = df["sector"].astype(str)
    return df


UNIVERSE = load_universe()


# ================================
# テクニカル指標
# ================================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
    except Exception:
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


def volume_ok(df: pd.DataFrame) -> bool:
    vol = df["Volume"].fillna(0)

    if len(vol) < 20:
        return False

    last_vol = float(vol.iloc[-1])
    if last_vol < MIN_VOLUME:
        return False

    avg20 = float(vol.tail(20).mean())
    avg5 = float(vol.tail(5).mean())
    avg2 = float(vol.tail(2).mean())

    # 押し目の理想パターン：静か→少し動きが出る
    cond1 = avg5 < avg20    # 直近5日で出来高が落ち着いている
    cond2 = avg2 > avg5     # 足元2日で少し戻っている

    return bool(cond1 and cond2)


def is_pullback(df: pd.DataFrame) -> bool:
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False

    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    prev_ma25 = safe_float(df["ma25"].iloc[-6])

    if not (np.isfinite(close) and np.isfinite(ma25) and np.isfinite(prev_ma25)):
        return False

    # 25MAより上（上昇トレンド中）
    if close < ma25:
        return False

    # 25MAが上向き
    if ma25 <= prev_ma25:
        return False

    # RSIゾーン（押し目）
    rsi = safe_float(last["rsi"])
    if not (RSI_MIN <= rsi <= RSI_MAX):
        return False

    # 出来高パターン
    if not volume_ok(df):
        return False

    return True


def calc_in_score(df: pd.DataFrame) -> int:
    """
    エントリー「しやすさ」を 0〜100点でスコアリング
    ※この後、地合い・セクターで微調整して Entry Edge にする
    """
    last = df.iloc[-1]
    rsi = safe_float(last["rsi"])
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    prev = safe_float(df.iloc[-2]["close"])

    score = 50

    # RSI評価（売られ具合）
    if np.isfinite(rsi):
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

    return int(max(0, min(score, 100)))


def calc_take_profit(df: pd.DataFrame) -> int:
    """
    直近高値と10日平均あたりから利確目安をざっくり推定
    """
    last = safe_float(df.iloc[-1]["close"])
    recent_high = safe_float(df["close"].tail(10).max())
    bb_mid = safe_float(df["ma10"].iloc[-1])

    if not np.isfinite(recent_high) or not np.isfinite(bb_mid):
        return int(last)

    tp = (recent_high * 0.6 + bb_mid * 0.4)
    return int(tp)


def calc_stop_loss(df: pd.DataFrame) -> int:
    """
    直近安値・25MA・現在値などから一番守りが固い損切りラインを選ぶ
    """
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


# ================================
# マクロ・市場サマリー部分
# ================================

def fetch_last_and_change(ticker: str, label: str, period: str = "5d") -> tuple[float, float]:
    """
    指数やETFの直近終値と1日騰落率を取得
    戻り値: (last, chg_pct) どちらか取れなければ (nan, nan)
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False
        )
    except Exception:
        return np.nan, np.nan

    if df is None or df.empty or "Close" not in df.columns or len(df) < 2:
        return np.nan, np.nan

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    if prev == 0:
        return last, np.nan

    chg = (last / prev - 1.0) * 100.0
    return last, chg


def calc_market_summary() -> dict:
    """
    グローバル指標から地合いスコアと
    『初心者でも読める短いサマリー文』を生成
    戻り値: dict(score:int, label:str, lines:list[str])
    """
    score = 50  # ベース
    lines: list[str] = []

    # 米株 ETF を指標として利用
    dia_last, dia_chg = fetch_last_and_change("DIA", "ダウ")
    qqq_last, qqq_chg = fetch_last_and_change("QQQ", "ナスダック100")
    iwm_last, iwm_chg = fetch_last_and_change("IWM", "ラッセル2000")
    soxx_last, soxx_chg = fetch_last_and_change("SOXX", "半導体")

    # VIX
    vix_last, vix_chg = fetch_last_and_change("^VIX", "VIX")

    # 米10年金利 (^TNX は10倍表記なので /10 前提)
    tnx_last, tnx_chg = fetch_last_and_change("^TNX", "米10年金利")

    # ドル円
    usdjpy_last, usdjpy_chg = fetch_last_and_change("JPY=X", "ドル円")

    # 欧州/アジア
    vkg_last, vkg_chg = fetch_last_and_change("VGK", "欧州株ETF")
    mchi_last, mchi_chg = fetch_last_and_change("MCHI", "中国株ETF")
    ewt_last, ewt_chg = fetch_last_and_change("EWT", "台湾株ETF")
    ewy_last, ewy_chg = fetch_last_and_change("EWY", "韓国株ETF")

    # コモディティ
    cl_last, cl_chg = fetch_last_and_change("CL=F", "原油先物")
    gc_last, gc_chg = fetch_last_and_change("GC=F", "金先物")
    hg_last, hg_chg = fetch_last_and_change("HG=F", "銅先物")

    # --- 米株全体の評価 ---
    us_moves = [dia_chg, qqq_chg, iwm_chg, soxx_chg]
    us_valid = [x for x in us_moves if np.isfinite(x)]
    us_avg = 0.0
    if us_valid:
        us_avg = sum(us_valid) / len(us_valid)
        # ±3% で ±15点くらいのイメージ
        score += max(-15, min(15, us_avg * 5))

    # --- VIX ---
    if np.isfinite(vix_last):
        if vix_last < 15:
            score += 10
        elif vix_last < 20:
            score += 0
        elif vix_last < 25:
            score -= 10
        else:
            score -= 20

    # --- 金利 ---
    yield10 = np.nan
    if np.isfinite(tnx_last):
        yield10 = tnx_last / 10.0  # ^TNX は10倍表記
        # ざっくり 4% 以下ならグロースにプラス、5%以上ならマイナス
        if yield10 < 4.0:
            score += 5
        elif yield10 > 5.0:
            score -= 5

    # --- 為替 ---
    if np.isfinite(usdjpy_last) and np.isfinite(usdjpy_chg):
        if usdjpy_chg > 0.5:
            score += 2
        elif usdjpy_chg < -0.5:
            score -= 2

    # スコアを 0〜100 にクリップ
    score = int(max(0, min(100, score)))

    # ラベル
    if score >= 70:
        label = "強い（リスクオン寄り）"
    elif score >= 60:
        label = "やや強い"
    elif score >= 45:
        label = "中立"
    elif score >= 30:
        label = "やや弱い"
    else:
        label = "弱い（調整モード）"

    # ============================
    # ここから「初心者向け・短いサマリー」を組み立てる
    # ============================

    # 1行目：米国株
    if np.isfinite(dia_chg) and np.isfinite(qqq_chg):
        if us_avg > 0.5:
            us_word = "強い上昇"
        elif us_avg > 0:
            us_word = "やや上昇"
        elif us_avg > -0.5:
            us_word = "小幅安"
        else:
            us_word = "弱い下げ"
        lines.append(
            f"- 米国株：ダウ {dia_chg:+.1f}% / ナスダック {qqq_chg:+.1f}% → 全体としては{us_word}"
        )
    else:
        lines.append("- 米国株：データ取得失敗（中立扱い）")

    # 2行目：半導体
    if np.isfinite(soxx_chg):
        sox_word = "追い風" if soxx_chg >= 0 else "逆風"
        lines.append(
            f"- 半導体ETF（SOXX）：{soxx_chg:+.1f}% → 日本のハイテクには{sox_word}"
        )
    else:
        lines.append("- 半導体ETF：データ取得失敗（影響は中立扱い）")

    # 3行目：金利
    if np.isfinite(yield10):
        gl_word = "プラス" if yield10 < 4.0 else ("やや逆風" if yield10 < 5.0 else "逆風")
        lines.append(
            f"- 米10年金利：{yield10:.2f}% → グロース株には{gl_word}"
        )
    else:
        lines.append("- 米10年金利：データ取得失敗（金利要因は中立扱い）")

    # 4行目：為替
    if np.isfinite(usdjpy_last) and np.isfinite(usdjpy_chg):
        fx_word = "外需に追い風" if usdjpy_chg >= 0 else "外需にはやや向かい風"
        lines.append(
            f"- 為替（ドル円）：{usdjpy_last:.1f}円（前日比 {usdjpy_chg:+.2f}%）→ {fx_word}"
        )
    else:
        lines.append("- 為替：データ取得失敗（中立扱い）")

    # 5行目：世界株ざっくり
    sub_parts = []
    if np.isfinite(vkg_chg):
        sub_parts.append(f"欧州 {vkg_chg:+.1f}%")
    if np.isfinite(mchi_chg):
        sub_parts.append(f"中国 {mchi_chg:+.1f}%")
    if np.isfinite(ewt_chg):
        sub_parts.append(f"台湾 {ewt_chg:+.1f}%")
    if np.isfinite(ewy_chg):
        sub_parts.append(f"韓国 {ewy_chg:+.1f}%")

    if sub_parts:
        lines.append("- 世界株： " + " / ".join(sub_parts))

    # 最後に一文まとめ
    lines.append(f"→ 今日の地合いスコア：{score}点（{label}）")

    return {
        "score": score,
        "label": label,
        "lines": lines
    }


def adjust_score_by_market(base_score: int, sector: str, market_score: int) -> int:
    """
    地合いスコアとセクターに応じて Entry Edge を微調整
    """
    score = base_score

    if market_score <= 40:
        # 弱地合い → ディフェンシブ加点、リスクセクター減点
        if sector in DEFENSIVE_SECTORS:
            score += 5
        if sector in RISK_SECTORS:
            score -= 10
    elif market_score >= 60:
        # 強地合い → ハイボラ・成長セクターを少し優遇
        if sector in RISK_SECTORS:
            score += 5
        if sector in DEFENSIVE_SECTORS:
            score -= 3

    return int(max(0, min(100, score)))


def pick_top5(market_score: int) -> list[dict]:
    rows: list[dict] = []

    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

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

        reasons: list[str] = []
        if np.isfinite(rsi):
            if rsi <= 32:
                reasons.append("売られすぎ気味の強い押し目")
            elif rsi <= 45:
                reasons.append("理想的な押し目ゾーン")
            else:
                reasons.append("浅めの押し目")

        ma25 = safe_float(last["ma25"])
        if np.isfinite(ma25) and ma25 != 0:
            dist25 = abs(price - ma25) / ma25
            if dist25 < 0.01:
                reasons.append("25日線タッチ")
            elif dist25 < 0.02:
                reasons.append("25日線近辺の押し目")

        if volume_ok(df):
            reasons.append("出来高が減少→直近でやや回復")

        base_score = calc_in_score(df)
        entry_edge = adjust_score_by_market(base_score, sector, market_score)
        tp = calc_take_profit(df)
        sl = calc_stop_loss(df)

        rows.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": price,
            "lower": lower,
            "rsi": rsi,
            "reason": " / ".join(reasons),
            "entry_edge": entry_edge,
            "tp": tp,
            "sl": sl
        })

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values("entry_edge", ascending=False).head(5)
    return df.to_dict("records")


# ================================
# メッセージ生成 & LINE送信
# ================================

def build_message() -> str:
    now_str = jst_now().strftime("%Y-%m-%d")

    # 1. 市場サマリー
    market = calc_market_summary()
    score = market["score"]
    summary_lines = market["lines"]

    # 2. 本命TOP5（地合いスコア連動）
    cands = pick_top5(score)

    if not cands:
        body = f"📉 {now_str}\n本日の本命Entry銘柄TOP5は条件に合致するものがありませんでした。"
        return body

    lines: list[str] = []

    # 市場サマリー
    lines.append("📊 今日の市場サマリー（初心者向け）")
    lines.extend(summary_lines)
    lines.append("")  # 空行

    # 個別銘柄
    lines.append(f"📈 {now_str} 本日の本命TOP5（押し目スイング候補）\n")

    for i, r in enumerate(cands, 1):
        lines.append(f"{i}. {r['ticker']}（{r['name']}）")
        lines.append(f"   Entry Edge（エントリー優位度）: {r['entry_edge']} / 100")
        lines.append(f"   買い目安: {r['lower']}円（現在値 約{int(r['price'])}円）")
        lines.append(f"   利確目安: {r['tp']}円")
        lines.append(f"   損切り目安: {r['sl']}円")
        if r["reason"]:
            lines.append(f"   根拠: {r['reason']}")
        lines.append("")  # 銘柄ごとに空行

    # まとめ
    lines.append("【まとめ：Entry Edgeランキング】")
    for r in cands:
        lines.append(f"- {r['ticker']}（{r['name']}）: Entry Edge {r['entry_edge']} / 100")

    body = "\n".join(lines)
    return body


def send_line(message: str) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("LINE_TOKENが設定されていません。以下のメッセージが本来送信されます：")
        print("------")
        print(message)
        print("------")
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


def main() -> None:
    """
    1. グローバル地合いをスコアリング
    2. 日本株ユニバースから押し目候補を抽出
    3. Entry EdgeでTOP5をランキング
    4. LINEにサマリー＋銘柄リストを送信
    """
    msg = build_message()
    print(msg)  # ログ確認用
    send_line(msg)


if __name__ == "__main__":
    main()