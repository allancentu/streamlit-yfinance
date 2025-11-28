import yfinance as yf
import pandas as pd
import datetime

def check_data():
    ticker = "AAPL"
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    
    # Fetch 1d with prepost=True to see the absolute latest data
    print("\n--- Period='1d' prepost=True ---")
    df = stock.history(period="1d", interval="1m", prepost=True)
    if df.empty:
        print("Empty.")
    else:
        print(df.tail())
        last_dt = df.index[-1]
        print(f"Last index: {last_dt} (tz: {last_dt.tzinfo})")
        
        # Check against current time
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        print(f"Current Time (UTC): {now_utc}")
        
        # Calculate age
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            last_dt = last_dt.astimezone(datetime.timezone.utc)
            
        age = (now_utc - last_dt).total_seconds() / 60
        print(f"Data Age: {age:.1f} minutes")

if __name__ == "__main__":
    check_data()
