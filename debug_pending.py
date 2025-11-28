import yfinance as yf
import pandas as pd
import datetime
from datetime import timedelta
import pytz

def debug_check(ticker_symbol):
    print(f"--- Debugging {ticker_symbol} ---")
    stock = yf.Ticker(ticker_symbol)
    
    # Fetch recent data to simulate a "last candle"
    hist = stock.history(period="1d", interval="1m")
    if hist.empty:
        print("No history found.")
        return

    # Pick a candle from a few minutes ago to simulate a pending prediction
    # Let's say 5 minutes ago
    if len(hist) < 10:
        print("Not enough history.")
        return
        
    last_candle_idx = -6
    original_time = hist.index[last_candle_idx]
    print(f"Original Time: {original_time} (tz: {original_time.tzinfo})")
    
    # Simulate storage as string
    time_str = str(original_time)
    print(f"Stored String: '{time_str}'")
    
    # Simulate retrieval
    last_candle_time = pd.to_datetime(time_str)
    print(f"Parsed Time: {last_candle_time} (tz: {last_candle_time.tzinfo})")
    
    # Simulate t+1 target
    target_time = last_candle_time + timedelta(minutes=1)
    print(f"Target Time (t+1): {target_time} (tz: {target_time.tzinfo})")
    
    # Now try to find this target_time in the history (simulating the fetch in check_prediction_result)
    # In the real app, we fetch again.
    df = stock.history(period="1d", interval="1m")
    
    # Logic from streamlit_app.py
    if df.index.tz is not None:
        df_index_utc = df.index.tz_convert('UTC')
    else:
        df_index_utc = df.index
        
    if target_time.tzinfo is not None:
        target_time_utc = target_time.astimezone(datetime.timezone.utc)
    else:
        target_time_utc = target_time
        
    print(f"Target Time UTC: {target_time_utc}")
    
    found = False
    for idx, row in df.iterrows():
        if idx.tzinfo is not None:
            idx_utc = idx.astimezone(datetime.timezone.utc)
        else:
            idx_utc = idx
            
        diff = (idx_utc - target_time_utc).total_seconds()
        
        # Print close matches
        if abs(diff) < 120:
            print(f"  Candidate: {idx_utc} (Diff: {diff}s)")
            
        if abs(diff) < 30:
            print(f"  -> MATCH FOUND! {idx_utc}")
            found = True
            break
            
    if not found:
        print("  -> NO MATCH FOUND.")

if __name__ == "__main__":
    debug_check("AAPL")
