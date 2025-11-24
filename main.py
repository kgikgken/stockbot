import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ==========================================
# 基本設定
# ==========================================

UNIVERSE_CSV_PATH = "universe_jpx.csv"
EARNINGS_CSV_PATH = "earnings_jpx.csv"
CREDIT_CSV_PATH = "credit_jpx.csv"

# ヒストリカル取得日数
HISTORY_PERIOD = "6mo"
MIN_HISTORY_DAYS = 60

# 流動性フィルター（最低売買代金）
MIN_AVG_TURNOVER = 3e8  # 3億円 / 日 目安

# ボラティリティ（ATR）フィルター
MAX_ATR_RATIO = 0.06        # ATR / Close 上限
MAX_ATR_MULTIPLE = 1.8      # ATR(14) <= 60日平均 * 1.8

# 押し目・RSI
RSI_MIN = 25
RSI_MAX = 40
MA_TOL = 0.01               # 25MA ±1%

# 出来高サイクル
VOLUME_SOLDOUT_RATIO = 0.4  # 30日平均の 0.4倍 未満 → 売り枯れ
VOLUME_SPIKE_RATIO = 2.2    # 30日平均の 2.2倍 超 → ニュース系スパイク

# リスクレジーム閾値（calc_market_summary の score を使用）
RISK_OFF_THRESHOLD = 40
RISK_ON_THRESHOLD = 60

# セクター分類（ざっくり）
DEFENSIVE_SECTORS = [
    "電気・ガス業", "食料品", "医薬品", "陸運業", "空運業",
    "小売業", "サービス業"
]
RISK_SECTORS = [
    "情報・通信業", "電気機器", "機械", "精密機器", "非鉄金属",
    "金属製品", "証券、商品先物取引業", "その他金融業"
]

# ==========================================
# ユーティリティ
# ==========================================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))

def safe_float(x, default=np.nan) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float(default)

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"])
    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)
    return df

def load_earnings() -> pd.DataFrame:
    if not os.path.exists(EARNINGS_CSV_PATH):
        print("WARN: earnings_jpx.csv が見つからないため、決算フィルターは無効（全通し）")
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    df = pd.read_csv(EARNINGS_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    else:
        df["earnings_date"] = pd.NaT
    return df

def load_credit() -> pd.DataFrame:
    if not os.path.exists(CREDIT_CSV_PATH):
        print("WARN: credit_jpx.csv が見つからないため、信用残フィルターは無効（全通し）")
        return pd.DataFrame(columns=["ticker", "margin_ratio", "margin_buy", "margin_sell"])
    df = pd.read_csv(CREDIT_CSV_PATH)
    df["ticker"] = df["ticker"].astype(str)
    for col in ["margin_ratio", "margin_buy", "margin_sell"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df

UNIVERSE = load_universe()
EARNINGS_DF = load_earnings()
CREDIT_DF = load_credit()

# ==========================================
# テクニカル指標
# ==========================================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["tr"] = tr
    df["atr"] = tr.rolling(period).mean()
    return df

def enrich_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close"] = df["Close"].astype(float)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["ma75"] = df["close"].rolling(75).mean()
    df = add_rsi(df, period=14)
    df = add_atr(df, period=14)
    df["turnover"] = df["close"] * df["Volume"].astype(float)
    return df

def fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.tail(120)
    if len(df) < MIN_HISTORY_DAYS:
        return None
    df = enrich_technicals(df)
    return df

# ==========================================
# ハードフィルター
# ==========================================

def passes_liquidity(df: pd.DataFrame) -> bool:
    recent = df.tail(20)
    avg_turnover = safe_float(recent["turnover"].mean())
    if not np.isfinite(avg_turnover):
        return False
    return avg_turnover >= MIN_AVG_TURNOVER

def passes_volatility(df: pd.DataFrame) -> bool:
    recent = df.tail(60).copy()
    if recent["atr"].isna().all():
        return False
    last = recent.iloc[-1]
    atr = safe_float(last["atr"])
    close = safe_float(last["close"])
    if not np.isfinite(atr) or not np.isfinite(close) or close <= 0:
        return False

    atr_ratio = atr / close
    if atr_ratio > MAX_ATR_RATIO:
        return False

    atr60 = safe_float(recent["atr"].mean())
    if np.isfinite(atr60) and atr > atr60 * MAX_ATR_MULTIPLE:
        return False

    return True

def passes_trend(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    ma10 = safe_float(last["ma10"])
    ma25 = safe_float(last["ma25"])
    ma75 = safe_float(last["ma75"])
    close = safe_float(last["close"])

    if not all(np.isfinite(v) for v in [ma10, ma25, ma75, close]):
        return False

    if not (ma10 >= ma25 >= ma75):
        return False
    if close < ma75:
        return False

    return True

def passes_event_risk(ticker: str, df: pd.DataFrame) -> bool:
    """
    決算日 ±3営業日を除外。
    earnings_jpx.csv にない銘柄はスルー。
    """
    if EARNINGS_DF.empty:
        return True

    sub = EARNINGS_DF[EARNINGS_DF["ticker"] == ticker]
    if sub.empty:
        return True

    earnings_date = sub["earnings_date"].iloc[0]
    if pd.isna(earnings_date):
        return True

    # df の最終日（直近営業日）
    last_date = df.index[-1].date()
    diff = (earnings_date - last_date).days
    # 決算 ±3日 は除外
    if -3 <= diff <= 3:
        return False

    return True

def passes_credit_risk(ticker: str, df: pd.DataFrame) -> bool:
    """
    信用倍率・信用買残の重さから危険銘柄を除外
    条件：
      - 信用倍率 <= 5
      - 信用買残 / 直近1週間出来高 <= 20
    """
    if CREDIT_DF.empty:
        return True

    sub = CREDIT_DF[CREDIT_DF["ticker"] == ticker]
    if sub.empty:
        return True

    row = sub.iloc[0]
    margin_ratio = safe_float(row.get("margin_ratio", np.nan))
    margin_buy = safe_float(row.get("margin_buy", np.nan))

    if np.isfinite(margin_ratio) and margin_ratio > 5.0:
        return False

    vol_week = safe_float(df["Volume"].tail(5).sum())
    if np.isfinite(margin_buy) and np.isfinite(vol_week) and vol_week > 0:
        buy_vs_vol = margin_buy / vol_week
        if buy_vs_vol > 20.0:
            return False

    return True

# ==========================================
# 出来高サイクル & 異常判定
# ==========================================

def analyze_volume_state(df: pd.DataFrame) -> dict:
    vol = df["Volume"].astype(float)
    if len(vol) < 30:
        return {
            "ok": False,
            "soldout": False,
            "spike": False,
        }

    v20 = safe_float(vol.tail(20).mean())
    v10 = safe_float(vol.tail(10).mean())
    v5 = safe_float(vol.tail(5).mean())
    last2 = safe_float(vol.tail(2).mean())
    v30 = safe_float(vol.tail(30).mean())
    last = safe_float(vol.iloc[-1])

    cond_cycle = v20 > v10 > v5 and last2 > v5

    soldout = np.isfinite(v30) and last < v30 * VOLUME_SOLDOUT_RATIO
    spike = np.isfinite(v30) and last > v30 * VOLUME_SPIKE_RATIO

    return {
        "ok": bool(cond_cycle and not spike),
        "soldout": bool(soldout),
        "spike": bool(spike),
    }

# ==========================================
# 押し目判定 & Entry Edge
# ==========================================

def is_deep_pullback(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])

    if not all(np.isfinite(v) for v in [close, ma25, rsi]):
        return False

    dist = abs(close - ma25) / ma25 if ma25 != 0 else np.inf
    if dist > MA_TOL:
        return False

    if not (RSI_MIN <= rsi <= RSI_MAX):
        return False

    return True

def analyze_candle(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    o = safe_float(last["Open"])
    h = safe_float(last["High"])
    l = safe_float(last["Low"])
    c = safe_float(last["Close"])

    body = abs(c - o)
    range_ = h - l
    lower_shadow = c - l if c > o else o - l

    long_lower = False
    if np.isfinite(range_) and range_ > 0 and np.isfinite(lower_shadow) and np.isfinite(body):
        if (lower_shadow / range_ > 0.35) and (lower_shadow > body):
            long_lower = True

    return {
        "long_lower": long_lower
    }

def calc_entry_edge(df: pd.DataFrame, volume_state: dict, candle: dict) -> tuple[int, list[str]]:
    last = df.iloc[-1]
    close = safe_float(last["close"])
    ma25 = safe_float(last["ma25"])
    rsi = safe_float(last["rsi"])

    score = 0
    reasons = []

    if passes_trend(df):
        score += 20
        reasons.append("上昇トレンド継続（10MA≥25MA≥75MA）")

    if np.isfinite(close) and np.isfinite(ma25) and ma25 > 0:
        dist = abs(close - ma25) / ma25
        if dist <= 0.005:
            score += 20
            reasons.append("25MAど真ん中の押し目")
        elif dist <= 0.01:
            score += 15
            reasons.append("25MA近辺の押し目")
        elif dist <= 0.02:
            score += 5
            reasons.append("25MA圏内")

    if np.isfinite(rsi):
        if RSI_MIN <= rsi <= 32:
            score += 25
            reasons.append("RSI深押しゾーン")
        elif 32 < rsi <= RSI_MAX:
            score += 15
            reasons.append("RSI押し目ゾーン")

    if volume_state["ok"]:
        score += 20
        reasons.append("出来高 減→増 の反転傾向")
    if volume_state["soldout"]:
        score += 5
        reasons.append("売り枯れ気味（出来高低水準）")
    if volume_state["spike"]:
        score -= 20
        reasons.append("出来高スパイク（ニュース系リスク）")

    if candle["long_lower"]:
        score += 10
        reasons.append("下ヒゲ反転気味")

    score = int(max(0, min(100, score)))
    return score, reasons

# ==========================================
# マクロ・地合い・テーマスコア
# ==========================================

def fetch_last_and_change(ticker: str, period: str = "5d") -> tuple[float, float]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return np.nan, np.nan

    if df is None or df.empty or "Close" not in df.columns or len(df) < 2:
        return np.nan, np.nan

    close = df["Close"].astype(float)
    last = safe_float(close.iloc[-1])
    prev = safe_float(close.iloc[-2])
    if not np.isfinite(last) or not np.isfinite(prev) or prev == 0:
        return np.nan, np.nan

    chg = (last / prev - 1.0) * 100.0
    return last, chg

def calc_market_summary() -> dict:
    lines = []
    score = 50

    dia_last, dia_chg = fetch_last_and_change("DIA")
    qqq_last, qqq_chg = fetch_last_and_change("QQQ")
    iwm_last, iwm_chg = fetch_last_and_change("IWM")
    soxx_last, soxx_chg = fetch_last_and_change("SOXX")

    vix_last, vix_chg = fetch_last_and_change("^VIX")
    tnx_last, tnx_chg = fetch_last_and_change("^TNX")
    usdjpy_last, usdjpy_chg = fetch_last_and_change("JPY=X")

    vkg_last, vkg_chg = fetch_last_and_change("VGK")
    mchi_last, mchi_chg = fetch_last_and_change("MCHI")
    ewt_last, ewt_chg = fetch_last_and_change("EWT")
    ewy_last, ewy_chg = fetch_last_and_change("EWY")

    us_moves = [dia_chg, qqq_chg, iwm_chg, soxx_chg]
    us_valid = [x for x in us_moves if np.isfinite(x)]
    if us_valid:
        us_avg = sum(us_valid) / len(us_valid)
        score += max(-15, min(15, us_avg * 5))
        lines.append(
            f"- 米株: ダウ {dia_chg:+.1f}％ / ナスダック100 {qqq_chg:+.1f}％ / ラッセル2000 {iwm_chg:+.1f}％ / 半導体SOXX {soxx_chg:+.1f}％"
        )
    else:
        lines.append("- 米株指標の取得に失敗（中立扱い）")

    if np.isfinite(vix_last):
        if vix_last < 15:
            score += 10
            vol_comment = "低ボラ（順張り有利）"
        elif vix_last < 20:
            vol_comment = "通常ボラ"
        elif vix_last < 25:
            score -= 10
            vol_comment = "やや高ボラ（慎重）"
        else:
            score -= 20
            vol_comment = "高ボラ（防御優先）"
        lines.append(f"- VIX {vix_last:.1f}（{vol_comment}）")
    else:
        lines.append("- VIX取得に失敗（ボラ要因は中立扱い）")

    if np.isfinite(tnx_last):
        y10 = tnx_last / 10.0
        if y10 < 4.0:
            score += 5
            tail = "グロース追い風"
        elif y10 > 5.0:
            score -= 5
            tail = "グロース逆風"
        else:
            tail = "金利中立"
        lines.append(f"- 米10年金利 {y10:.2f}％（{tail}）")
    else:
        lines.append("- 米10年金利取得に失敗（金利要因は中立）")

    if np.isfinite(usdjpy_last) and np.isfinite(usdjpy_chg):
        lines.append(
            f"- ドル円 {usdjpy_last:.1f}円（{usdjpy_chg:+.2f}％）、外需/輸出に{'追い風' if usdjpy_chg > 0 else '逆風気味'}"
        )

    asia_eu = []
    if np.isfinite(vkg_chg):
        asia_eu.append(f"欧州 {vkg_chg:+.1f}％")
    if np.isfinite(mchi_chg):
        asia_eu.append(f"中国 {mchi_chg:+.1f}％")
    if np.isfinite(ewt_chg):
        asia_eu.append(f"台湾 {ewt_chg:+.1f}％")
    if np.isfinite(ewy_chg):
        asia_eu.append(f"韓国 {ewy_chg:+.1f}％")
    if asia_eu:
        lines.append("- 欧州・アジア: " + " / ".join(asia_eu))

    score = int(max(0, min(100, score)))

    if score >= RISK_ON_THRESHOLD:
        label = "強め（リスクオン寄り）"
        regime = "risk_on"
    elif score <= RISK_OFF_THRESHOLD:
        label = "弱め（リスクオフ寄り）"
        regime = "risk_off"
    else:
        label = "中立〜レンジ"
        regime = "neutral"

    lines.append(f"→ 今日の地合いスコア: {score}点（{label}）")

    return {
        "score": score,
        "label": label,
        "lines": lines,
        "regime": regime,
    }

def calc_theme_score(sector: str, market: dict) -> int:
    base = 50
    regime = market["regime"]

    if sector in RISK_SECTORS:
        if regime == "risk_on":
            base += 20
        elif regime == "risk_off":
            base -= 10
    if sector in DEFENSIVE_SECTORS:
        if regime == "risk_off":
            base += 15
        elif regime == "risk_on":
            base -= 5

    base = int(max(0, min(100, base)))
    return base

def calc_market_fit(sector: str, market: dict) -> int:
    regime = market["regime"]
    if regime == "risk_on":
        if sector in RISK_SECTORS:
            return 80
        if sector in DEFENSIVE_SECTORS:
            return 50
        return 60
    elif regime == "risk_off":
        if sector in DEFENSIVE_SECTORS:
            return 80
        if sector in RISK_SECTORS:
            return 40
        return 50
    else:
        if sector in DEFENSIVE_SECTORS:
            return 60
        if sector in RISK_SECTORS:
            return 60
        return 55

def decide_risk_regime_action(market: dict) -> dict:
    regime = market["regime"]

    if regime == "risk_off":
        return {
            "regime_label": "守り優先",
            "max_leverage": 1.2,
            "max_positions": 2,
            "comment": "新規スイングは慎重。既存ポジの縮小・リスク管理優先。",
        }
    elif regime == "risk_on":
        return {
            "regime_label": "攻め寄り",
            "max_leverage": 2.0,
            "max_positions": 4,
            "comment": "押し目スイングには追い風。条件の揃った銘柄だけレバを乗せてよい。",
        }
    else:
        return {
            "regime_label": "中立",
            "max_leverage": 1.5,
            "max_positions": 3,
            "comment": "軽めのスイングは可。ただしイベント前は無理に攻める場面ではない。",
        }

# ==========================================
# 銘柄スクリーニング
# ==========================================

def classify_core_watch(entry_edge: int, hard_pass: bool) -> str | None:
    if not hard_pass:
        return None

    if entry_edge >= 75:
        return "core"
    if entry_edge >= 60:
        return "watch"
    return None

def calc_final_rank(entry_edge: int, theme_score: int, market_fit: int) -> float:
    return entry_edge * 0.55 + theme_score * 0.30 + market_fit * 0.15

def calc_take_profit(df: pd.DataFrame) -> int:
    last = safe_float(df["close"].iloc[-1])
    recent_high = safe_float(df["close"].tail(10).max())
    ma10 = safe_float(df["ma10"].iloc[-1])

    if not np.isfinite(recent_high) or not np.isfinite(ma10):
        return int(last) if np.isfinite(last) else 0

    tp = recent_high * 0.6 + ma10 * 0.4
    return int(tp)

def calc_stop_loss(df: pd.DataFrame) -> int:
    last = safe_float(df["close"].iloc[-1])
    ma25 = safe_float(df["ma25"].iloc[-1])
    recent_low = safe_float(df["close"].tail(5).min())

    candidates = []
    if np.isfinite(recent_low):
        candidates.append(recent_low)
    if np.isfinite(ma25):
        candidates.append(ma25 * 0.985)
    if np.isfinite(last):
        candidates.append(last * 0.97)

    if not candidates:
        return int(last) if np.isfinite(last) else 0

    return int(min(candidates))

def screen_candidates(market: dict) -> tuple[list[dict], list[dict]]:
    core_rows = []
    watch_rows = []

    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = row["sector"]

        df = fetch_history(ticker)
        if df is None:
            continue

        if not passes_liquidity(df):
            continue
        if not passes_volatility(df):
            continue
        if not passes_trend(df):
            continue
        if not is_deep_pullback(df):
            continue
        if not passes_event_risk(ticker, df):
            continue
        if not passes_credit_risk(ticker, df):
            continue

        volume_state = analyze_volume_state(df)
        candle = analyze_candle(df)
        entry_edge, reasons_edge = calc_entry_edge(df, volume_state, candle)

        hard_pass = True

        class_type = classify_core_watch(entry_edge, hard_pass)
        if class_type is None:
            continue

        theme_score = calc_theme_score(sector, market)
        market_fit = calc_market_fit(sector, market)
        final_rank = calc_final_rank(entry_edge, theme_score, market_fit)

        last = df.iloc[-1]
        price = safe_float(last["close"])
        ma5 = safe_float(last["ma5"])
        ma10 = safe_float(last["ma10"])
        ma25 = safe_float(last["ma25"])

        lower_candidates = [v for v in [ma5, ma10, ma25] if np.isfinite(v)]
        if lower_candidates:
            buy_low = int(min(lower_candidates))
            buy_high = int(max(lower_candidates))
        else:
            buy_low = buy_high = int(price)

        tp = calc_take_profit(df)
        sl = calc_stop_loss(df)

        reasons = []
        if is_deep_pullback(df):
            reasons.append("深めの押し目（RSI & 25MA）")
        reasons.extend(reasons_edge)
        if volume_state["soldout"]:
            reasons.append("売り枯れ気味")
        if volume_state["spike"]:
            reasons.append("出来高スパイク注意")

        rec = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "class": class_type,
            "entry_edge": entry_edge,
            "theme_score": theme_score,
            "market_fit": market_fit,
            "final_rank": final_rank,
            "price": int(price) if np.isfinite(price) else 0,
            "buy_low": buy_low,
            "buy_high": buy_high,
            "tp": tp,
            "sl": sl,
            "reasons": " / ".join(sorted(set(reasons))),
        }

        if class_type == "core":
            core_rows.append(rec)
        else:
            watch_rows.append(rec)

    core_rows = sorted(core_rows, key=lambda x: x["final_rank"], reverse=True)[:4]
    watch_rows = sorted(watch_rows, key=lambda x: x["final_rank"], reverse=True)[:5]

    return core_rows, watch_rows

# ==========================================
# メッセージ組み立て
# ==========================================

def build_message() -> str:
    today = jst_now().strftime("%Y-%m-%d")

    market = calc_market_summary()
    risk_cfg = decide_risk_regime_action(market)
    core, watch = screen_candidates(market)

    lines: list[str] = []

    lines.append(f"📅 {today} stockbot TOM 戦略レポート")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(
        f"- 地合い: {market['score']}点（{market['label']} / {risk_cfg['regime_label']}）"
    )
    lines.append(
        f"- 推奨レバレッジ上限: 約 {risk_cfg['max_leverage']:.1f}倍 / 最大ポジ数: {risk_cfg['max_positions']}銘柄"
    )
    lines.append(f"- 戦略コメント: {risk_cfg['comment']}")
    lines.append("")

    lines.append("◆ マクロ・地合いサマリー")
    lines.extend(market["lines"])
    lines.append("")

    lines.append("◆ 今日の戦い方（スイング視点）")
    if market["regime"] == "risk_off":
        lines.append("- 原則守り。新規スイングは「本命」でもサイズは半分以下。")
        lines.append("- 決算・イベント跨ぎは避ける。含み益は部分利確を優先。")
    elif market["regime"] == "risk_on":
        lines.append("- 条件の整った押し目には素直に乗る日。")
        lines.append("- とはいえ、レバは上限内。飛びつきは厳禁。")
    else:
        lines.append("- 軽めに攻めてもよいが、イベント前の全力はNG。")
        lines.append("- 押し目以外（高値追い・逆張り）はスルー推奨。")
    lines.append("")

    if core:
        lines.append("◆ 本命（Core）候補")
        for i, r in enumerate(core, 1):
            lines.append(f"{i}. {r['ticker']}（{r['name']} / {r['sector']}）")
            lines.append(f"   Entry Edge: {r['entry_edge']} / 100")
            lines.append(
                f"   買いゾーン: {r['buy_low']}〜{r['buy_high']}円（現在 {r['price']}円）"
            )
            lines.append(f"   利確目安: {r['tp']}円 / 損切り目安: {r['sl']}円")
            lines.append(
                f"   テーマ・地合い適合: Theme {r['theme_score']} / MarketFit {r['market_fit']}"
            )
            lines.append(f"   コメント: {r['reasons']}")
            lines.append("")
    else:
        lines.append("◆ 本命（Core）候補")
        lines.append("- 条件を満たす本命押し目は本日なし。無理にポジションを取りに行かない方が合理的。")
        lines.append("")

    if watch:
        lines.append("◆ 注目（Watch）候補")
        for i, r in enumerate(watch, 1):
            lines.append(f"{i}. {r['ticker']}（{r['name']} / {r['sector']}）")
            lines.append(
                f"   Entry Edge: {r['entry_edge']} / テーマ: {r['theme_score']} / MarketFit: {r['market_fit']}"
            )
            lines.append(
                "   状況: 押し目仕上がり途中の候補。板・寄り付きの動き次第で本命化を検討。"
            )
            lines.append("")
    else:
        lines.append("◆ 注目（Watch）候補")
        lines.append("- 本日時点で“仕上がり途中”の押し目候補も少数。様子見優位の地合い。")
        lines.append("")

    lines.append("◆ まとめ")
    if core:
        core_tick = ", ".join([f"{r['ticker']}({r['entry_edge']}点)" for r in core])
        lines.append(f"- 本命: {core_tick}")
    else:
        lines.append("- 本命: なし（今日は無理に攻める日ではない）")
    if watch:
        watch_tick = ", ".join([r["ticker"] for r in watch])
        lines.append(f"- 注目: {watch_tick}")
    else:
        lines.append("- 注目: なし")

    lines.append("")
    lines.append("Only the edge. Nothing else.")

    return "\n".join(lines)

# ==========================================
# LINE 送信
# ==========================================

def send_line(message: str) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("LINE_TOKEN が設定されていないため、標準出力にメッセージを出します。")
        print(message)
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {"messages": [{"type": "text", "text": message}]}

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print("LINE status:", resp.status_code)
        if resp.status_code != 200:
            print("LINE response:", resp.text)
    except Exception as e:
        print("LINE送信エラー:", e)

# ==========================================
# メイン
# ==========================================

def main():
    msg = build_message()
    print(msg)
    send_line(msg)

if __name__ == "__main__":
    main()