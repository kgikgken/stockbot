import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
import xml.etree.ElementTree as ET

# =============================
# 基本設定 / Config
# =============================

UNIVERSE_CSV_PATH = "universe_jpx.csv"

# 押し目ロジック
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

# 強化 W 流動性フィルタ閾値
MIN_AVG_VOLUME = 200_000        # 5日平均出来高 20万株
MIN_DAILY_VALUE = 500_000_000   # 売買代金 5億円

# ニュース評価用キーワード
SECTOR_NEWS_KEYWORDS = {
    "石油・石炭製品": "石油 セクター",
    "医薬品": "医薬品 セクター",
    "海運業": "海運 セクター",
    "鉱業": "鉱業 セクター",
    "陸運業": "陸運 セクター",
    # 必要に応じて追加
}

POSITIVE_WORDS = ["増益", "上方修正", "最高益", "好調", "堅調", "続伸", "買い", "急騰"]
NEGATIVE_WORDS = ["減益", "下方修正", "悪化", "下落", "急落", "売り", "軟調"]

# 材料タグ用キーワード
MATERIAL_KEYWORDS = {
    "決算": ["決算", "四半期", "通期", "業績"],
    "上方修正": ["上方修正", "増額修正"],
    "下方修正": ["下方修正", "減額修正"],
    "増配・株主還元": ["増配", "配当", "自社株買い", "株主還元"],
    "提携・M&A": ["提携", "協業", "合弁", "M&A", "買収", "資本業務提携"],
    "AI・DX": ["AI", "生成AI", "DX", "デジタル"],
    "新製品・サービス": ["新製品", "新サービス", "発売", "発表", "投入"],
    "設備投資・増産": ["設備投資", "新工場", "増産", "生産能力", "建設"],
    "不祥事・行政処分": ["行政処分", "業務停止", "不祥事", "不正", "改善命令", "検査"],
}

# LINE 文字数
MAX_LINE_TEXT_LEN = 3900

# ニュースキャッシュ
NEWS_CACHE: Dict[str, Dict[str, object]] = {}


# =============================
# ユーティリティ
# =============================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def safe_float(x) -> float:
    """Series や NaN が来ても float に変換"""
    if isinstance(x, pd.Series):
        if not len(x):
            return float("nan")
        x = x.iloc[-1]
    try:
        return float(x)
    except Exception:
        return float("nan")


# =============================
# ユニバース読み込み
# =============================

def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_CSV_PATH):
        raise FileNotFoundError(f"{UNIVERSE_CSV_PATH} が見つかりません")

    df = pd.read_csv(UNIVERSE_CSV_PATH)
    df = df.dropna(subset=["ticker", "name", "sector"]).copy()

    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)
    if "industry_big" in df.columns:
        df["industry_big"] = df["industry_big"].astype(str)
    if "market" in df.columns:
        df["market"] = df["market"].astype(str)

    return df


UNIVERSE_DF = load_universe()
TICKER_NAME: Dict[str, str] = dict(zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["name"]))
TICKER_SECTOR: Dict[str, str] = dict(zip(UNIVERSE_DF["ticker"], UNIVERSE_DF["sector"]))


# =============================
# ニューススコア & 材料要約
# =============================

def fetch_sector_news_info(sector: str) -> Tuple[float, List[str]]:
    """
    そのセクターのニューススコアと材料タグ一覧を返す。
    （Google News RSS を使用。1回取得したら NEWS_CACHE に保存）
    """
    if sector in NEWS_CACHE:
        info = NEWS_CACHE[sector]
        return info["score"], info["materials"]

    try:
        keyword = SECTOR_NEWS_KEYWORDS.get(sector, sector)
        query = quote(keyword + " 株")
        url = (
            "https://news.google.com/rss/search?"
            f"q={query}&hl=ja&gl=JP&ceid=JP:ja"
        )

        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            NEWS_CACHE[sector] = {"score": 0.0, "materials": []}
            return 0.0, []

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        if not items:
            NEWS_CACHE[sector] = {"score": 0.0, "materials": []}
            return 0.0, []

        score = 0.0
        material_counts = {k: 0 for k in MATERIAL_KEYWORDS.keys()}

        for item in items:
            title_raw = item.findtext("title", default="")
            title = str(title_raw)

            # ポジ・ネガ判定
            for w in POSITIVE_WORDS:
                if w in title:
                    score += 1.0
            for w in NEGATIVE_WORDS:
                if w in title:
                    score -= 1.0

            # 材料タグ
            for tag, words in MATERIAL_KEYWORDS.items():
                if any(word in title for word in words):
                    material_counts[tag] += 1

        score /= max(len(items), 1)

        sorted_materials = sorted(
            [k for k, v in material_counts.items() if v > 0],
            key=lambda k: material_counts[k],
            reverse=True,
        )

        NEWS_CACHE[sector] = {"score": float(score), "materials": sorted_materials}
        return float(score), sorted_materials

    except Exception as e:
        print(f"[WARN] ニュース取得失敗: sector={sector} / {e}")
        NEWS_CACHE[sector] = {"score": 0.0, "materials": []}
        return 0.0, []


# =============================
# テクニカル指標
# =============================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        df["atr"] = np.nan
        return df

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    df["atr"] = atr
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        df["vwap"] = np.nan
        return df

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].fillna(0).astype(float)

    typical_price = (high + low + close) / 3.0
    cum_vol = vol.cumsum()
    cum_tp_vol = (typical_price * vol).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap"] = vwap
    return df


# =============================
# データ取得 & 加工
# =============================

def fetch_history(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    yfinance から60営業日分を取得。
    成功時は DataFrame、失敗時は None。
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
        print(f"[WARN] yfinance ダウンロード失敗: {ticker} / {e}")
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
    df = add_atr(df)
    df = add_vwap(df)

    return df


# =============================
# 出来高パターン判定
# =============================

def volume_pattern_ok(df: pd.DataFrame) -> bool:
    if "Volume" not in df.columns:
        return False

    vol = df["Volume"].fillna(0)
    if len(vol) < 20:
        return False

    avg20 = float(vol.tail(20).mean())
    avg5 = float(vol.tail(5).mean())
    avg2 = float(vol.tail(2).mean())

    cond_decrease = avg5 < avg20
    cond_increase = avg2 > avg5

    return bool(cond_decrease and cond_increase)


# =============================
# 強化版 W流動性フィルター
# =============================

def is_liquid(df: pd.DataFrame,
              min_volume: int = MIN_AVG_VOLUME,
              min_value: int = MIN_DAILY_VALUE) -> bool:
    if "Volume" not in df.columns:
        return False

    vol = df["Volume"].fillna(0)
    if len(vol) < 5:
        return False

    avg5 = float(vol.tail(5).mean())
    if avg5 < min_volume:
        return False

    last = df.iloc[-1]
    price = safe_float(last.get("close"))
    today_vol = safe_float(last.get("Volume"))
    if not (np.isfinite(price) and np.isfinite(today_vol)):
        return False

    value = price * today_vol
    if value < min_value:
        return False

    return True


# =============================
# 押し目判定ロジック
# =============================

def is_uptrend(df: pd.DataFrame) -> bool:
    if len(df) < 30:
        return False

    last = df.iloc[-1]
    close_now = safe_float(last["close"])
    ma25_now = safe_float(last["ma25"])
    ma25_prev = safe_float(df["ma25"].iloc[-6])

    if not np.isfinite(close_now) or not np.isfinite(ma25_now) or not np.isfinite(ma25_prev):
        return False

    if close_now < ma25_now:
        return False
    if ma25_now <= ma25_prev:
        return False

    return True


def is_near_ma(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    close_now = safe_float(last["close"])
    ma5 = safe_float(last["ma5"])
    ma10 = safe_float(last["ma10"])

    if not np.isfinite(close_now):
        return False

    cond_ma5 = np.isfinite(ma5) and abs(close_now - ma5) / ma5 <= PULLBACK_MA_TOL
    cond_ma10 = np.isfinite(ma10) and abs(close_now - ma10) / ma10 <= PULLBACK_MA_TOL
    return bool(cond_ma5 or cond_ma10)


def is_rsi_ok(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    rsi = safe_float(last.get("rsi", np.nan))
    return bool(np.isfinite(rsi) and RSI_MIN <= rsi <= RSI_MAX)


def is_volume_turn(df: pd.DataFrame) -> bool:
    return volume_pattern_ok(df)


def is_pullback(df: pd.DataFrame) -> bool:
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return False
    if not is_liquid(df):
        return False

    return all(
        [
            is_uptrend(df),
            is_near_ma(df),
            is_rsi_ok(df),
            is_volume_turn(df),
        ]
    )


# =============================
# 買いレンジ計算（精密・下限寄り）
# =============================

def calc_buy_range(df: pd.DataFrame) -> Tuple[float, float]:
    """
    MA5 / MA10 / VWAP / ATR を使って
    下限寄りの狭い買いレンジを計算
    """
    last = df.iloc[-1]

    price = safe_float(last["close"])
    ma5 = safe_float(last.get("ma5", np.nan))
    ma10 = safe_float(last.get("ma10", np.nan))
    vwap = safe_float(last.get("vwap", np.nan))
    atr = safe_float(last.get("atr", np.nan))

    base_candidates = [v for v in [ma5, ma10, vwap, price] if np.isfinite(v)]
    base = float(np.mean(base_candidates)) if base_candidates else price

    if np.isfinite(atr) and atr > 0:
        # ATR の0.7〜0.3部分をレンジにする（下限寄り）
        buy_lower = base - 0.7 * atr
        buy_upper = base - 0.3 * atr
    else:
        if base_candidates:
            buy_lower = min(base_candidates)
            buy_upper = max(base_candidates)
        else:
            buy_lower = price
            buy_upper = price

    if buy_lower > buy_upper:
        buy_lower, buy_upper = buy_upper, buy_lower

    return buy_lower, buy_upper


# =============================
# セクター強度計算（ニュース込み）
# =============================

def calc_sector_strength() -> pd.DataFrame:
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
            if not np.isfinite(close_now):
                continue

            if len(df) >= 6:
                base_close = safe_float(df["close"].iloc[-6])
                ma25_prev = safe_float(df["ma25"].iloc[-6])
            else:
                base_close = safe_float(df["close"].iloc[0])
                ma25_prev = safe_float(df["ma25"].iloc[0])

            if base_close <= 0 or not np.isfinite(base_close):
                continue

            ret_1d_raw = last.get("ret_1d", np.nan)
            ret_1d = float(safe_float(ret_1d_raw) * 100)
            ret_5d = float((close_now / base_close - 1) * 100)

            ma25_now = safe_float(last.get("ma25", np.nan))
            if np.isfinite(ma25_now) and np.isfinite(ma25_prev) and ma25_prev != 0:
                slope25_val = (ma25_now - ma25_prev) / ma25_prev * 100
            else:
                slope25_val = 0.0
            slope25 = float(slope25_val)

            if (
                not np.isfinite(ret_1d)
                or not np.isfinite(ret_5d)
                or not np.isfinite(slope25)
            ):
                continue

            vals.append((ret_1d, ret_5d, slope25))

        if not vals:
            continue

        arr = np.array(vals, dtype=float)
        avg_1d = float(arr[:, 0].mean())
        avg_5d = float(arr[:, 1].mean())
        avg_slope25 = float(arr[:, 2].mean())

        news_score, materials = fetch_sector_news_info(sector)
        total_score = avg_5d * 0.6 + avg_slope25 * 0.3 + news_score * 0.5

        # 材料トピックは一番多く出ているもの1つだけ
        top_material = materials[0] if materials else ""

        records.append(
            {
                "sector": sector,
                "avg_1d": avg_1d,
                "avg_5d": avg_5d,
                "avg_slope25": avg_slope25,
                "news_score": float(news_score),
                "material": top_material,
                "total_score": float(total_score),
            }
        )

    return pd.DataFrame(records)


# =============================
# 相場地合いスコア（N225 / TOPIX）
# =============================

def calc_market_regime() -> Dict[str, Dict[str, float]]:
    indices = {
        "日経平均": "^N225",
        "TOPIX ETF": "1306.T",  # TOPIX連動ETF
    }

    result: Dict[str, Dict[str, float]] = {}

    for name, ticker in indices.items():
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

        if base_close <= 0 or not np.isfinite(base_close):
            continue

        ret_1d = float(safe_float(last.get("ret_1d", np.nan)) * 100)
        ret_5d = float((close_now / base_close - 1) * 100)

        ma25_now = safe_float(last.get("ma25", np.nan))
        if np.isfinite(ma25_now) and np.isfinite(ma25_prev) and ma25_prev != 0:
            slope25 = float((ma25_now - ma25_prev) / ma25_prev * 100)
        else:
            slope25 = 0.0

        score = ret_5d * 0.6 + slope25 * 0.4

        result[name] = {
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "slope25": slope25,
            "score": score,
        }

    return result


def describe_market_score(score: float) -> str:
    if score >= 1.0:
        return "かなり強気"
    if score >= 0.4:
        return "強気"
    if score >= -0.2:
        return "中立"
    if score >= -0.8:
        return "弱気"
    return "かなり弱気"


# =============================
# ヒートマップ用ヘルパー
# =============================

def _sector_heat_emoji(ret_5d: float) -> str:
    """5日騰落率からセクターヒートマップ用の色を決める"""
    if ret_5d >= 2.0:
        return "🟩"  # 強い上昇
    elif ret_5d >= 0.5:
        return "🟨"  # やや上昇
    elif ret_5d >= -0.5:
        return "🟦"  # 横ばい〜小動き
    elif ret_5d >= -2.0:
        return "🟧"  # やや下落
    else:
        return "🟥"  # 下落


def _stock_heat_emoji(chg_1d: float) -> str:
    """日中変化率から銘柄ヒートマップ用の色を決める"""
    if chg_1d >= 3.0:
        return "🟩"
    elif chg_1d >= 1.0:
        return "🟨"
    elif chg_1d >= -1.0:
        return "🟦"
    elif chg_1d >= -3.0:
        return "🟧"
    else:
        return "🟥"


# =============================
# 候補抽出
# =============================

def pick_candidates_in_sector(strong_sectors: List[str]) -> pd.DataFrame:
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

        buy_lower, buy_upper = calc_buy_range(df)

        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "price": price,
                "chg_1d": chg_1d,
                "rsi": rsi,
                "buy_lower": buy_lower,
                "buy_upper": buy_upper,
            }
        )

    return pd.DataFrame(rows)


def pick_candidates_outside_sector(strong_sectors: List[str]) -> pd.DataFrame:
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

        buy_lower, buy_upper = calc_buy_range(df)

        ma25 = safe_float(last["ma25"])
        if ma25 > 0:
            ma25_dis = abs(price - ma25) / ma25
        else:
            ma25_dis = np.nan

        vol = df["Volume"].dropna()
        if len(vol) >= 6:
            recent5 = vol.tail(6).iloc[:-1]
            vol_score = safe_float(vol.iloc[-1] / recent5.mean())
        else:
            vol_score = 1.0

        score = (
            (rsi if np.isfinite(rsi) else 100.0) * WEIGHT_RSI
            + (ma25_dis if np.isfinite(ma25_dis) else 1.0) * WEIGHT_MA25
            + (-vol_score) * WEIGHT_VOLUME
        )

        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "price": price,
                "chg_1d": chg_1d,
                "rsi": rsi,
                "buy_lower": buy_lower,
                "buy_upper": buy_upper,
                "score": score,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("score")  # スコアが低いほど良い押し目

    return df


# =============================
# 買いレンジ表示フォーマット（下限のみ）
# =============================

def _format_buy_text(low: float, high: float) -> str:
    """買いレンジ表示：下限のみ。「円付近」も付けない"""
    if not np.isfinite(low):
        return "-"
    return f"{int(low)}円"


def _format_candidates_table(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("銘柄 | 買いレンジ")
    lines.append("---- | ----")
    for _, r in df.iterrows():
        txt = _format_buy_text(safe_float(r["buy_lower"]), safe_float(r["buy_upper"]))
        lines.append(
            f"{r['ticker']}（{r['name']}） | {txt}"
        )
    return lines


# =============================
# メッセージ生成
# =============================

def build_message() -> str:
    # 地合い
    market = calc_market_regime()

    # セクター強度
    sec_df = calc_sector_strength()
    if sec_df.empty:
        return "セクター情報が取得できませんでした。"

    sec_df = sec_df.sort_values("total_score", ascending=False)
    top = sec_df.head(TOP_SECTOR_COUNT).reset_index(drop=True)
    strong_sectors = list(top["sector"])

    now = jst_now()
    lines: List[str] = []
    lines.append(f"📈 {now:%Y-%m-%d} スイング候補レポート\n")

    # --- 相場地合いスコア ---
    if market:
        lines.append("【相場地合いスコア】")
        total = 0.0
        n = 0
        for name, vals in market.items():
            desc = describe_market_score(vals["score"])
            lines.append(
                f"- {name}: 1日 {vals['ret_1d']:.1f}% / 5日 {vals['ret_5d']:.1f}% / "
                f"25日線傾き {vals['slope25']:.2f}% → {desc}"
            )
            total += vals["score"]
            n += 1
        if n:
            overall = describe_market_score(total / n)
            lines.append(f"⇒ 地合い総合評価: {overall}\n")

    # --- セクター強度 ---
    lines.append("【今日のテーマ候補（セクターベース）】")
    for _, r in top.iterrows():
        comment = ""
        if r["avg_5d"] > 0 and r["avg_slope25"] > 0 and r["news_score"] > 0:
            comment = "（ニュース追い風の強い上昇トレンド）"
        elif r["avg_5d"] > 0 and r["avg_slope25"] > 0:
            comment = "（強い上昇トレンド）"
        elif r["avg_5d"] > 0:
            comment = "（短期強め）"
        elif r["avg_slope25"] > 0:
            comment = "（押し目セクター）"
        else:
            comment = "（相対的にマシ）"

        lines.append(
            f"- {r['sector']}: 1日 {r['avg_1d']:.1f}% / "
            f"5日 {r['avg_5d']:.1f}% / 25日線傾き {r['avg_slope25']:.2f}% / "
            f"ニュース {r['news_score']:.2f} {comment}"
        )

    # --- セクターヒートマップ ---
    lines.append("\n【セクターヒートマップ（5日騰落率）】")
    for _, r in sec_df.iterrows():
        emoji = _sector_heat_emoji(r["avg_5d"])
        lines.append(f"{emoji} {r['sector']}: {r['avg_5d']:.1f}%")

    # --- 主な材料トピック ---
    lines.append("\n【主な材料トピック（上位セクター）】")
    for _, r in top.iterrows():
        mat = str(r.get("material", "")).strip()
        if mat:
            lines.append(f"- {r['sector']}: {mat}")

    # --- TOP5セクター内銘柄 ---
    cands_in = pick_candidates_in_sector(strong_sectors)

    lines.append("\n【押し目スイング候補（TOP5セクター内）】")
    if cands_in.empty:
        lines.append("条件に合う銘柄がありません。")
    else:
        for sector, grp in cands_in.groupby("sector"):
            lines.append(f"▼{sector}")
            lines.extend(_format_candidates_table(grp))

    # --- セクター外候補（ACDE複合スコア） ---
    cands_out = pick_candidates_outside_sector(strong_sectors)

    lines.append("\n【セクター外おすすめ押し目銘柄】")
    if cands_out.empty:
        lines.append("セクター外では押し目優良銘柄なし。")
    else:
        for sector, grp in cands_out.groupby("sector"):
            lines.append(f"▼{sector}")
            lines.extend(_format_candidates_table(grp))

    # --- 候補銘柄ヒートマップ ---
    if not cands_in.empty or not cands_out.empty:
        lines.append("\n【候補銘柄ヒートマップ（日中変化率）】")
        try:
            df_all = []
            if not cands_in.empty:
                df_all.append(cands_in[["ticker", "name", "chg_1d"]].copy())
            if not cands_out.empty:
                df_all.append(cands_out[["ticker", "name", "chg_1d"]].copy())
            all_df = pd.concat(df_all, ignore_index=True)

            # 多すぎると見づらいので上位30銘柄まで
            for _, r in all_df.head(30).iterrows():
                chg = safe_float(r["chg_1d"])
                emoji = _stock_heat_emoji(chg)
                lines.append(
                    f"{emoji} {r['ticker']}（{r['name']}）: {chg:.1f}%"
                )
        except Exception as e:
            print("[WARN] ヒートマップ生成中にエラー:", e)

    return "\n".join(lines)


# =============================
# LINE送信
# =============================

def _split_message(text: str, limit: int = MAX_LINE_TEXT_LEN) -> List[str]:
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            parts.append(current.rstrip())
            current = ""
        current += line + "\n"

    if current.strip():
        parts.append(current.rstrip())

    return parts


def send_line(message: str) -> None:
    token = os.getenv("LINE_TOKEN")
    if not token:
        print("[ERROR] LINE_TOKEN が設定されていません")
        return

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    chunks = _split_message(message)

    for i in range(0, len(chunks), 5):  # 1リクエスト最大5メッセージ
        batch = chunks[i: i + 5]
        data = {"messages": [{"type": "text", "text": t} for t in batch]}

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            print(f"[INFO] LINE API status: {resp.status_code}")
            if resp.status_code != 200:
                print("[ERROR] LINE API response:", resp.text)
        except Exception as e:
            print("[ERROR] LINE送信中に例外が発生:", e)


# =============================
# main()
# =============================

def main() -> None:
    try:
        msg = build_message()
    except Exception as e:
        error_msg = f"スクリーニング中にエラーが発生しました: {e}"
        print("[ERROR] build_message 失敗:", e)
        msg = error_msg

    send_line(msg)


if __name__ == "__main__":
    main()