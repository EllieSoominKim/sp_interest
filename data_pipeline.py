import pandas as pd
import yfinance as yf #Yahoo Finance 자동 다운로드
import certifi, os 

os.environ["SSL_CERT_FILE"] = certifi.where() #SSL 인증서 참조(yfinance)

def load_data(start="2014-01-01", end="2024-01-01"):
    # -----SPY ETF 다운로드-----
    sp500 = yf.download("SPY", start=start, end=end, auto_adjust=True)
    if sp500.empty:
        raise ValueError("SPY 다운로드 실패")
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0) #다중 계층 형태면 첫번째 수준
    sp500 = sp500[["Close"]] #종가
    sp500.rename(columns={"Close": "sp500"}, inplace=True)

    #-----금리Rate-----
    rate = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS")
    rate.rename(columns={rate.columns[0]: "Date", rate.columns[1]: "Rate"}, inplace=True)
    rate["Date"] = pd.to_datetime(rate["Date"])
    rate.set_index("Date", inplace=True)
    rate = rate.resample("D").ffill() #하루 단위 리셈플링, 결측값은 forward-fill

    #-----병합-----
    df = sp500.join(rate, how="inner") #일별 기준
    df.dropna(inplace=True) 

    # -----파생 변수-----
    df["return"] = df["sp500"].pct_change() #(오늘 종가 - 어제 종가) / 어제 종가
    df["volatility"] = df["return"].rolling(20).std() #20일 이동 표준 편차
    df["rate_change"] = df["Rate"].diff() #오늘 금리 - 어제 금리
    df.dropna(inplace=True)

    return df


