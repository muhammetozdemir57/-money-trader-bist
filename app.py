import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime

st.set_page_config(
    page_title="Borsa Radar",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1.5rem;
}
.metric-card {
    background: #161b22;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #30363d;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #00ff99;
}
.sub-title {
    color: #c9d1d9;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📈 Borsa Radar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Yapay Zekâ Destekli BIST Tarama ve Teknik Analiz Platformu</div>', unsafe_allow_html=True)
st.divider()

BIST_FALLBACK = [
    "AKBNK", "ASELS", "ASTOR", "BIMAS", "EREGL", "GARAN", "HEKTS",
    "ISCTR", "KCHOL", "KOZAL", "KRDMD", "MIATK", "PATEK", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "THYAO", "TOASO", "TUPRS",
    "YKBNK", "ZOREN", "FROTO", "TAVHL", "TCELL", "TTKOM", "YEOTK",
    "SMRTG", "KONTR", "GESAN", "EUPWR", "CWENE", "ODAS", "ALARK",
    "ENKAI", "OYAKC", "CIMSA", "BRSAN", "BRYAT", "MGROS", "MAVI",
    "DOAS", "ENJSA", "AKSEN", "ALFAS", "KCAER", "KONYA", "OTKAR",
    "SDTTR", "FORTE", "RTALB", "REEDR", "TABGD", "TAVHL"
]


def clean_symbol(symbol):
    symbol = str(symbol).upper().strip()
    symbol = symbol.replace(".IS", "")
    symbol = symbol.replace("BIST:", "")
    symbol = "".join(ch for ch in symbol if ch.isalnum())
    return symbol


@st.cache_data(ttl=3600)
def get_bist_symbols():
    try:
        url = "https://www.oyakyatirim.com.tr/piyasa-verileri/XUTUM"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            verify=False
        )

        tables = pd.read_html(response.text)
        symbols = []

        for table in tables:
            for col in table.columns:
                col_name = str(col).lower()
                if "sembol" in col_name or "kod" in col_name:
                    symbols.extend(table[col].dropna().astype(str).tolist())

        cleaned = []
        for s in symbols:
            s = clean_symbol(s)
            if 3 <= len(s) <= 6:
                cleaned.append(s)

        cleaned = sorted(list(set(cleaned)))

        if len(cleaned) > 100:
            return cleaned

    except Exception:
        pass

    return sorted(list(set(BIST_FALLBACK)))


@st.cache_data(ttl=900)
def get_stock_data(symbol, period="6mo"):
    try:
        ticker = symbol + ".IS"
        df = yf.Ticker(ticker).history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df[df["Volume"] > 0]

        if len(df) < 50:
            return None

        return df

    except Exception:
        return None


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def calc_obv(df):
    values = [0]

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            values.append(values[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            values.append(values[-1] - df["Volume"].iloc[i])
        else:
            values.append(values[-1])

    return pd.Series(values, index=df.index)


def calc_macd(close):
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    macd = ema12 - ema26
    signal = calc_ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def analyze_stock(symbol, df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    prev_high = float(high.iloc[-2])
    prev_low = float(low.iloc[-2])

    rsi_val = float(calc_rsi(close).iloc[-1])

    ema20 = float(calc_ema(close, 20).iloc[-1])
    ema50 = float(calc_ema(close, 50).iloc[-1])
    ema200 = float(calc_ema(close, 200).iloc[-1]) if len(df) >= 200 else ema50

    atr_val = float(calc_atr(df).iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = last_close * 0.02

    vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

    obv_val = calc_obv(df)
    obv_ema = calc_ema(obv_val, 3)
    obv_buy = obv_val.iloc[-1] > obv_ema.iloc[-1]

    macd, macd_signal, macd_hist = calc_macd(close)
    macd_positive = macd.iloc[-1] > macd_signal.iloc[-1]
    macd_momentum = macd_hist.iloc[-1] > macd_hist.iloc[-2]

    breakout_up = last_close > prev_high and prev_close <= prev_high
    breakout_down = last_close < prev_low and prev_close >= prev_low

    trend_bull = last_close > ema20 > ema50
    trend_strong = last_close > ema20 > ema50 > ema200
    price_above_ema20 = last_close > ema20

    volume_strong = vol_ratio >= 1.3
    volume_explosive = vol_ratio >= 2.0

    rsi_good = 45 <= rsi_val <= 70
    rsi_low_turn = 30 <= rsi_val < 45
    rsi_hot = rsi_val > 70

    score = 0

    if trend_bull:
        score += 18

    if trend_strong:
        score += 12

    if price_above_ema20:
        score += 8

    if breakout_up:
        score += 18

    if volume_strong:
        score += 12

    if volume_explosive:
        score += 8

    if obv_buy:
        score += 10

    if macd_positive:
        score += 8

    if macd_momentum:
        score += 6

    if rsi_good:
        score += 10

    if rsi_low_turn:
        score += 6

    if rsi_hot:
        score -= 5

    score = int(max(0, min(score, 100)))

    tavan_score = 0

    if breakout_up:
        tavan_score += 3
    if volume_explosive:
        tavan_score += 3
    elif volume_strong:
        tavan_score += 2
    if trend_bull:
        tavan_score += 2
    if obv_buy:
        tavan_score += 1
    if macd_positive:
        tavan_score += 1
    if 50 <= rsi_val <= 75:
        tavan_score += 1

    tavan_score = min(tavan_score, 10)

    target_1 = last_close + atr_val * 2
    target_2 = last_close + atr_val * 3.5
    stop_loss = last_close - atr_val * 1.5

    expected_pct = ((target_1 - last_close) / last_close) * 100
    risk_pct = ((last_close - stop_loss) / last_close) * 100

    if score >= 85:
        signal = "🔥 ÇOK GÜÇLÜ AL"
    elif score >= 70:
        signal = "✅ GÜÇLÜ AL"
    elif score >= 55:
        signal = "🟡 TAKİP"
    elif breakout_down:
        signal = "🔻 SAT"
    else:
        signal = "⚪ ZAYIF"

    if tavan_score >= 8:
        tavan_status = "🔥 YÜKSEK"
    elif tavan_score >= 6:
        tavan_status = "🟡 ORTA"
    else:
        tavan_status = "⚪ DÜŞÜK"

    return {
        "Sembol": symbol,
        "Fiyat": round(last_close, 2),
        "Sinyal": signal,
        "Skor": score,
        "Tavan Puanı": tavan_score,
        "Tavan İhtimali": tavan_status,
        "RSI": round(rsi_val, 2),
        "Hacim x": round(vol_ratio, 2),
        "MACD": "Pozitif" if macd_positive else "Negatif",
        "OBV": "AL" if obv_buy else "Zayıf",
        "Trend": "Güçlü" if trend_strong else "Pozitif" if trend_bull else "Zayıf",
        "Kırılım": "Yukarı" if breakout_up else "Aşağı" if breakout_down else "Yok",
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "Hedef 1": round(target_1, 2),
        "Hedef 2": round(target_2, 2),
        "Stop": round(stop_loss, 2),
        "Beklenti %": round(expected_pct, 2),
        "Risk %": round(risk_pct, 2)
    }


symbols = get_bist_symbols()

with st.sidebar:
    st.header("⚙️ Radar Ayarları")

    st.info(f"Bulunan sembol sayısı: {len(symbols)}")

    scan_count = st.slider(
        "Taranacak hisse sayısı",
        min_value=10,
        max_value=len(symbols),
        value=min(100, len(symbols)),
        step=10
    )

    period = st.selectbox(
        "Veri periyodu",
        ["3mo", "6mo", "1y"],
        index=1
    )

    min_score = st.slider(
        "Minimum skor filtresi",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

    only_buy = st.checkbox("Sadece AL / TAKİP sinyalleri", value=False)

    run_scan = st.button("🔍 BIST RADAR TARAMASINI BAŞLAT", use_container_width=True)


col1, col2, col3, col4 = st.columns(4)

col1.metric("📋 Sembol", len(symbols))
col2.metric("📊 Periyot", period)
col3.metric("🎯 Min Skor", min_score)
col4.metric("🔍 Tarama", scan_count)

st.divider()

if run_scan:
    selected_symbols = symbols[:scan_count]
    results = []

    progress = st.progress(0)
    status = st.empty()

    start_time = time.time()

    for i, symbol in enumerate(selected_symbols):
        status.write(f"⏳ {symbol} taranıyor...")

        df = get_stock_data(symbol, period)

        if df is not None:
            try:
                result = analyze_stock(symbol, df)
                results.append(result)
            except Exception:
                pass

        progress.progress((i + 1) / len(selected_symbols))
        time.sleep(0.05)

    elapsed = time.time() - start_time

    status.write("✅ Tarama tamamlandı.")

    if results:
        data = pd.DataFrame(results)
        data = data.sort_values(by="Skor", ascending=False)

        data = data[data["Skor"] >= min_score]

        if only_buy:
            data = data[data["Sinyal"].isin(["🔥 ÇOK GÜÇLÜ AL", "✅ GÜÇLÜ AL", "🟡 TAKİP"])]

        st.success(f"Tarama tamamlandı. Süre: {elapsed:.1f} saniye | Sonuç: {len(data)} hisse")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("🔥 En Yüksek Skor", int(data["Skor"].max()) if len(data) else 0)
        m2.metric("✅ AL Sinyali", len(data[data["Skor"] >= 70]))
        m3.metric("🔥 Çok Güçlü AL", len(data[data["Skor"] >= 85]))
        m4.metric("🚀 Tavan Adayı", len(data[data["Tavan Puanı"] >= 7]))

        st.subheader("🔥 En Güçlü Radar Sonuçları")
        st.dataframe(data, use_container_width=True, hide_index=True)

        st.subheader("🚀 Tavan Radar")
        tavanlar = data[data["Tavan Puanı"] >= 7]

        if len(tavanlar):
            st.dataframe(tavanlar, use_container_width=True, hide_index=True)
        else:
            st.warning("Şu an güçlü tavan adayı bulunamadı.")

        st.subheader("✅ Güçlü AL Listesi")
        strong_buy = data[data["Skor"] >= 70]

        if len(strong_buy):
            st.dataframe(strong_buy, use_container_width=True, hide_index=True)
        else:
            st.warning("Güçlü AL sinyali bulunamadı.")

        csv = data.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📥 Sonuçları CSV indir",
            data=csv,
            file_name=f"borsa_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.error("Veri alınamadı veya sonuç üretilemedi.")

else:
    st.warning("Sol menüden ayarları seçip 'BIST RADAR TARAMASINI BAŞLAT' butonuna bas.")

st.divider()
st.caption("⚠️ Bu uygulama yatırım tavsiyesi değildir. Eğitim ve teknik analiz amaçlıdır.")
