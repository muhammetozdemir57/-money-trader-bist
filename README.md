# 💰 Money Trader BIST

Borsa İstanbul hisseleri için Streamlit tabanlı teknik analiz ve tarama uygulaması.

## Özellikler

- BIST hisselerini tarar
- RSI hesaplar
- EMA 20 / EMA 50 / EMA 200 kontrol eder
- OBV analizi yapar
- ATR ile hedef ve stop seviyesi üretir
- Hacim artışını kontrol eder
- Kırılım sinyali arar
- Skor sistemiyle hisseleri sıralar
- Tavan adayı olabilecek hisseleri listeler

## Kullanılan Teknolojiler

- Python
- Streamlit
- Pandas
- NumPy
- yfinance
- Plotly
- TA

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
