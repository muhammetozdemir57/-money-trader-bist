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

st.title("📈 BORSA RADAR")
st.caption("Yapay Zekâ Destekli BIST Tarama ve Teknik Analiz Platformu")

BIST_FALLBACK = [
    "AKBNK", "ASELS", "ASTOR", "BIMAS", "EREGL", "GARAN", "HEKTS",
    "ISCTR", "KCHOL", "KOZAL", "KRDMD", "MIATK", "PATEK", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "THYAO", "TOASO", "TUPRS",
    "YKBNK", "ZOREN", "FROTO", "TAVHL", "TCELL", "TTKOM", "YEOTK",
    "SMRTG", "KONTR", "GESAN", "EUPWR", "CWENE", "ODAS", "ALARK",
    "ENKAI", "OYAKC", "CIMSA", "BRSAN", "BRYAT", "MGROS", "MAVI"
]


@st.cache_data(ttl=3600)
def get_bist_symbols():
    try:
        url = "https://www.oyakyatirim.com.tr/piyasa-verileri/XUTUM"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        tables = pd.read_html(r.text)

        symbols = []

        for table in tables:
            for col in table.columns:
                if "Sembol" in str(col) or "Kod" in str(col):
                    symbols += table[col].dropna().astype(str).tolist()

        clean = []
        for s in symbols:
            s = s.upper().replace(".IS", "").replace("BIST:", "").strip()
            s = "".join(ch for ch in s if ch.isalnum())
            if 3 <= len(s) <= 6:
                clean.append(s)

        clean = sorted(list(set(clean)))

        if len(clean) > 100:
            return clean

    except Exception:
        pass

    return sorted(list(set(BIST_FALLBACK)))


@st.cache_data(ttl=900)
def get_data(symbol, period="6mo"):
    try:
        df = yf.Ticker(symbol + ".IS").history(period=period, interval="1d")

        if df is None or df.empty or len(df) < 50:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df[df["Volume"] > 0]

        if len(df) < 50:
            return None

        return df

    except Exception:
        return None


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def obv(df):
    values = [0]

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            values.append(values[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            values.append(values[-1] - df["Volume"].iloc[i])
        else:
            values.append(values[-1])

    return pd.Series(values, index=df.index)


def analyze(symbol, df):
    close = df["Close"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])

    rsi_val = rsi(close).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1] if len(df) >= 200 else ema50

    atr_val = atr(df).iloc[-1]
    vol_ma = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1

    obv_val = obv(df)
    obv_ema = ema(obv_val, 3)

    breakout_up = last_close > prev_high
    breakout_down = last_close < prev_low

    trend_bull = last_close > ema20 > ema50
    trend_strong = last_close > ema20 > ema50 > ema200

    obv_buy = obv_val.iloc[-1] > obv_ema.iloc[-1]
    volume_strong = vol_ratio >= 1.3
    rsi_good = 45 <= rsi_val <= 70

    score = 0

    if trend_bull:
        score += 20

    if trend_strong:
        score += 15

    if breakout_up:
        score += 20

    if volume_strong:
        score += 15

    if obv_buy:
        score += 10

    if rsi_good:
        score += 10

    if last_close > ema20:
        score += 10

    score = min(score, 100)

    target = last_close + atr_val * 2 if not np.isnan(atr_val) else last_close * 1.05
    stop = last_close - atr_val * 1.5 if not np.isnan(atr_val) else last_close * 0.97

    if score >= 80:
        signal = "🔥 ÇOK GÜÇLÜ AL"
    elif score >= 65:
        signal = "✅ AL"
    elif score >= 50:
        signal = "🟡 TAKİP"
    elif breakout_down:
        signal = "🔻 SAT"
    else:
        signal = "⚪ ZAYIF"

    tavan_score = 0

    if breakout_up:
        tavan_score += 3
    if volume_strong:
        tavan_score += 2
    if trend_bull:
        tavan_score += 2
    if obv_buy:
        tavan_score += 1
    if rsi_val < 75:
        tavan_score += 1

    return {
        "Sembol": symbol,
        "Fiyat": round(last_close, 2),
        "Sinyal": signal,
        "Skor": score,
        "Tavan Puanı": tavan_score,
        "RSI": round(rsi_val, 2),
        "Hacim x": round(vol_ratio, 2),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "Hedef": round(target, 2),
        "Stop": round(stop, 2),
        "Kırılım": "Yukarı" if breakout_up else "Aşağı" if breakout_down else "Yok"
    }


symbols = get_bist_symbols()

st.sidebar.header("⚙️ Ayarlar")

max_stock = st.sidebar.slider(
    "Taranacak hisse sayısı",
    min_value=10,
    max_value=len(symbols),
    value=min(100, len(symbols)),
    step=10
)

period = st.sidebar.selectbox(
    "Veri periyodu",
    ["3mo", "6mo", "1y"],
    index=1
)

run = st.sidebar.button("🔍 Taramayı Başlat")

st.info(f"Toplam sembol bulundu: {len(symbols)}")

if run:
    results = []
    progress = st.progress(0)
    status = st.empty()

    selected_symbols = symbols[:max_stock]

    for i, symbol in enumerate(selected_symbols):
        status.write(f"⏳ {symbol} taranıyor...")

        df = get_data(symbol, period)

        if df is not None:
            result = analyze(symbol, df)
            results.append(result)

        progress.progress((i + 1) / len(selected_symbols))
        time.sleep(0.05)

    if results:
        data = pd.DataFrame(results)
        data = data.sort_values(by="Skor", ascending=False)

        st.success(f"Tarama tamamlandı. {len(data)} hisse analiz edildi.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("En yüksek skor", int(data["Skor"].max()))
        col2.metric("AL sinyali", len(data[data["Skor"] >= 65]))
        col3.metric("Çok güçlü AL", len(data[data["Skor"] >= 80]))
        col4.metric("Tavan adayı", len(data[data["Tavan Puanı"] >= 7]))

        st.subheader("🔥 En Güçlü Hisseler")
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🚀 Tavan Adayları")
        tavanlar = data[data["Tavan Puanı"] >= 7]

        if len(tavanlar) > 0:
            st.dataframe(
                tavanlar,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Şu an tavan adayı bulunamadı.")

        csv = data.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📥 Sonuçları CSV indir",
            data=csv,
            file_name=f"money_trader_bist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    else:
        st.error("Veri alınamadı veya sonuç üretilemedi.")

else:
    st.warning("Sol menüden tarama sayısını seçip 'Taramayı Başlat' butonuna bas.")
