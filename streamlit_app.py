import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import mplfinance as mpf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import io
import time
from PIL import Image
from datetime import datetime as dt, timedelta
import pytz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import do_network from stock_cnn.py
# We need to make sure stock_cnn.py is in the path or just copy the function if it's easier
# Since it's in the same directory, we can try to import it.
# However, stock_cnn.py has a lot of top-level code that runs on import.
# To avoid running that code, it's safer to copy the do_network function here or wrap it in stock_cnn.py
# Given the constraints, I will copy the necessary parts of do_network here to ensure stability.

def do_network(cdl_columns, dropout_rate=0.35):
    """
    Cria um modelo de saída dupla para análise de padrões de candlestick.
    Copied from stock_cnn.py to avoid side effects on import.
    """
    num_cdl_patterns = len(cdl_columns)
    
    inputs = tf.keras.layers.Input(shape=(128,128,3), name='input_image')

    # Bloco 1: 32 filtros
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)
    s2 = tf.keras.layers.Dropout(dropout_rate)(s2)

    # Bloco 2: 64 filtros
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(s2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    s4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)
    s4 = tf.keras.layers.Dropout(dropout_rate)(s4)

    # Bloco 3: 128 filtros
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(s4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c5)
    s6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c5)
    s6 = tf.keras.layers.Dropout(dropout_rate)(s6)

    # Achata características da CNN
    flat = tf.keras.layers.Flatten()(s6)

    # Camadas densas para processamento de características
    f7 = tf.keras.layers.Dense(256, activation='relu')(flat)
    f7 = tf.keras.layers.BatchNormalization()(f7)
    f7 = tf.keras.layers.Dropout(dropout_rate)(f7)
    f8 = tf.keras.layers.Dense(128, activation='relu')(f7)
    f8 = tf.keras.layers.BatchNormalization()(f8)
    f8 = tf.keras.layers.Dropout(dropout_rate)(f8)

    # Saída 1: Predições de Padrões CDL
    cdl_patterns = tf.keras.layers.Dense(
        num_cdl_patterns,
        activation='sigmoid',
        name='cdl_patterns'
    )(f8)

    # Cabeça MLP: Predições de Direção de Preço
    mlp_hidden = tf.keras.layers.Dense(128, activation='relu')(cdl_patterns)
    mlp_hidden = tf.keras.layers.BatchNormalization()(mlp_hidden)
    mlp_hidden = tf.keras.layers.Dropout(dropout_rate)(mlp_hidden)

    # Saída 2: Predições de Direção de Preço
    price_directions = tf.keras.layers.Dense(
        6,
        activation='sigmoid',
        name='price_directions'
    )(mlp_hidden)

    return tf.keras.models.Model(
        inputs=inputs,
        outputs=[cdl_patterns, price_directions],
        name='cnn_mlp_dual_output'
    )

@st.cache_resource
def load_model():
    """Load the trained model with weights."""
    # Dummy columns to define architecture (assuming 20 patterns as per default)
    # We label them generically since we don't have the original metadata
    dummy_cdl_columns = [f"Pattern_{i}" for i in range(20)]
    
    model = do_network(dummy_cdl_columns)
    try:
        model.load_weights('model/best_model.weights.h5')
        return model, dummy_cdl_columns
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None, None

def generate_prediction_image(data):
    """Generate a 128x128 candlestick image from data."""
    # Normalize data for plotting (similar to training)
    # Note: mpf.plot handles scaling, but we need to ensure the style matches training
    # Training used: style='yahoo', volume=True/False (random), no axes
    
    buf = io.BytesIO()
    
    # Create a copy to avoid modifying original
    plot_data = data.copy()
    
    # Plot configuration to match training as closely as possible
    # Training code: figratio=(1, 1), figsize=(3, 3)
    
    # Custom style to remove grid and axes if needed, but 'yahoo' is standard
    # We need to remove axes manually after plotting or use returnfig=True
    
    fig, axes = mpf.plot(
        plot_data,
        type='candle',
        style='yahoo',
        volume=True,
        figratio=(1, 1),
        figsize=(3, 3),
        returnfig=True
    )
    
    # Remove axes/ticks to match training data
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            ax.set_ylim(ymin - 0.5, ymax + 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.axis('off') # Turn off axis completely
        
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    img = Image.open(buf)
    
    # Convert to RGB (remove alpha if present)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize to 128x128 as expected by model
    img = img.resize((128, 128))
    
    return img

    return img

def check_prediction_result(prediction, ticker_symbol):
    """
    Verify if a prediction was correct based on subsequent market data.
    Returns the updated prediction dictionary.
    """
    # Parse timestamps
    try:
        # We stored the last candle time as a string, need to parse it back
        if 'Last Candle Time' not in prediction:
            return prediction
            
        # Parse the stored time
        time_str = prediction['Last Candle Time']
        if isinstance(time_str, str) and time_str.endswith(" ET"):
            time_str = time_str[:-3]
            last_candle_time = pd.to_datetime(time_str)
            # It was ET, so localize if naive
            if last_candle_time.tzinfo is None:
                last_candle_time = pytz.timezone('America/New_York').localize(last_candle_time)
        else:
            last_candle_time = pd.to_datetime(time_str)
        
        initial_close = prediction['Initial Close']
        
        # Ensure last_candle_time is timezone-aware and convert to UTC for internal logic
        if last_candle_time.tzinfo is None:
            # If naive, assume UTC to be safe, or try to infer? 
            # Ideally we should have stored it with timezone.
            last_candle_time = last_candle_time.replace(tzinfo=datetime.timezone.utc)
        else:
            last_candle_time = last_candle_time.astimezone(datetime.timezone.utc)
        
        # Define horizons in minutes
        horizons = {
            't+1': 1,
            't+5': 5,
            't+30': 30
        }
        
        stock = yf.Ticker(ticker_symbol)
        
        # Check each horizon
        for horizon_name, minutes in horizons.items():
            result_key = f"{horizon_name} Result"
            pred_key = f"{horizon_name} Prediction"
            
            # If already verified, skip
            if prediction[result_key] in ["✅ Correct", "❌ Incorrect", "Neutral"]:
                continue
                
            target_time_utc = last_candle_time + timedelta(minutes=minutes)
            
            # Current time (UTC)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            
            # Define wait times (when to allow checking)
            wait_delays = {
                't+1': 2,
                't+5': 6,
                't+30': 31
            }
            
            check_time_utc = target_time_utc + timedelta(minutes=1)
            
            # If current time is before the check time, tell user to wait
            if check_time_utc > now_utc:
                # Convert to Eastern Time for display
                et_tz = pytz.timezone('America/New_York')
                check_time_et = check_time_utc.astimezone(et_tz)
                wait_time_str = check_time_et.strftime("%H:%M ET")
                prediction[result_key] = f"⏳ Wait until {wait_time_str}"
                continue
            
            # Target time is in the past, try to fetch data
            try:
                # Fetch 1m data for the last day, INCLUDING extended hours
                df = stock.history(period="1d", interval="1m", prepost=True)
                
                # Check if data is fresh enough
                if not df.empty:
                    last_dt = df.index[-1]
                    # Convert to UTC
                    if last_dt.tzinfo is None:
                        last_dt_utc = last_dt.replace(tzinfo=datetime.timezone.utc)
                    else:
                        last_dt_utc = last_dt.astimezone(datetime.timezone.utc)
                        
                    # If the latest data is older than our target, we definitely need more data
                    if last_dt_utc < target_time_utc:
                        # Try fetching 5d to be sure
                        df = stock.history(period="5d", interval="1m", prepost=True)

                if df.empty:
                    df = stock.history(period="5d", interval="1m", prepost=True)
                
                if df.empty:
                    prediction[result_key] = f"⏳ No data yet..."
                    continue
                
                # Convert df index to UTC
                if df.index.tz is not None:
                    df_index_utc = df.index.tz_convert('UTC')
                else:
                    # If naive, assume UTC (or ET? yfinance usually returns ET with offset or UTC)
                    # Let's assume the index is correct as provided by yfinance
                    df_index_utc = df.index.tz_localize('UTC')
                
                # Find the candle at target_time_utc
                target_candle = None
                
                # Robust search
                for idx, row in df.iterrows():
                    # idx is already in df_index_utc (if we iterated over that) 
                    # but iterrows gives original index.
                    
                    if idx.tzinfo is not None:
                        idx_utc = idx.astimezone(datetime.timezone.utc)
                    else:
                        idx_utc = idx.replace(tzinfo=datetime.timezone.utc)
                        
                    # Check match
                    if abs((idx_utc - target_time_utc).total_seconds()) < 30:
                        target_candle = row
                        break
                
                if target_candle is None:
                    # Debug info
                    latest_data_time = df.index[-1]
                    if latest_data_time.tzinfo is not None:
                        latest_utc = latest_data_time.astimezone(datetime.timezone.utc)
                    else:
                        latest_utc = latest_data_time.replace(tzinfo=datetime.timezone.utc)
                    
                    # Convert to ET for display
                    et_tz = pytz.timezone('America/New_York')
                    latest_et = latest_utc.astimezone(et_tz)
                    latest_str = latest_et.strftime('%H:%M ET')
                    
                    # If we are way past the target time and still no data, maybe market closed?
                    if (now_utc - target_time_utc).total_seconds() > 3600: # 1 hour past
                         prediction[result_key] = f"⚠️ Market Closed? (Latest: {latest_str})"
                    else:
                         prediction[result_key] = f"⏳ Data pending... (Latest: {latest_str})"
                    continue
                
                target_close = float(target_candle['Close'])
                
                # Determine actual movement
                if target_close > initial_close:
                    actual_direction = "Up"
                elif target_close < initial_close:
                    actual_direction = "Down"
                else:
                    actual_direction = "Neutral"
                
                # Compare with prediction
                predicted_direction = prediction[pred_key]
                
                if predicted_direction == actual_direction:
                    prediction[result_key] = "✅ Correct"
                else:
                    prediction[result_key] = "❌ Incorrect"
                
                # Store actual direction for metrics calculation
                prediction[f"{horizon_name} Actual"] = actual_direction
                    
            except Exception as e:
                prediction[result_key] = "Error checking"
                print(f"Error checking result for {horizon_name}: {e}")
                
    except Exception as e:
        print(f"Error in check_prediction_result: {e}")
        
    return prediction

    return prediction

def calculate_metrics(predictions):
    """Calculate comprehensive performance metrics."""
    metrics_data = []
    
    # Containers for Multi-label metrics (only rows where ALL horizons are verified)
    multilabel_rows_true = []
    multilabel_rows_pred = []
    
    # Containers for Per-Horizon metrics (all verified outcomes for that horizon)
    horizon_true = {1: [], 5: [], 30: []}
    horizon_pred = {1: [], 5: [], 30: []}
    
    for pred in predictions:
        # Check if this row is fully verified for multi-label
        row_true = []
        row_pred = []
        fully_verified = True
        
        for k in [1, 5, 30]:
            actual = pred.get(f"t+{k} Actual")
            predicted = pred.get(f"t+{k} Prediction")
            result = pred.get(f"t+{k} Result")
            
            if result in ["✅ Correct", "❌ Incorrect"] and actual and predicted:
                horizon_true[k].append(actual)
                horizon_pred[k].append(predicted)
                row_true.append(actual)
                row_pred.append(predicted)
            else:
                fully_verified = False
        
        if fully_verified:
            multilabel_rows_true.append(row_true)
            multilabel_rows_pred.append(row_pred)

    # 1. Standard Summary Metrics (Global & Per Horizon)
    def calc_summary(y_true, y_pred, label):
        if not y_true: return None
        return {
            "Horizon": label,
            "Samples": len(y_true),
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 (Weighted)": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    # Global lists for summary
    global_true = [item for sublist in horizon_true.values() for item in sublist]
    global_pred = [item for sublist in horizon_pred.values() for item in sublist]
    
    if global_true:
        metrics_data.append(calc_summary(global_true, global_pred, "Global (All)"))
    for k in [1, 5, 30]:
        if horizon_true[k]:
            metrics_data.append(calc_summary(horizon_true[k], horizon_pred[k], f"t+{k}"))
            
    summary_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()

    # 2. Confusion Matrices & Class Reports
    confusion_matrices = {}
    class_reports = {}
    
    labels = ["Up", "Down", "Neutral"] # Standardize labels
    
    for k in [1, 5, 30]:
        if horizon_true[k]:
            # Confusion Matrix
            # Use labels to ensure fixed size even if some classes are missing
            cm = confusion_matrix(horizon_true[k], horizon_pred[k], labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
            confusion_matrices[k] = cm_df
            
            # Class Report
            report = classification_report(horizon_true[k], horizon_pred[k], labels=labels, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            class_reports[k] = report_df

    # 3. Multi-label Metrics
    multilabel_metrics = {}
    if multilabel_rows_true:
        # Exact Match Ratio: Fraction of rows where row_true == row_pred
        exact_matches = sum([1 for i in range(len(multilabel_rows_true)) if multilabel_rows_true[i] == multilabel_rows_pred[i]])
        exact_match_ratio = exact_matches / len(multilabel_rows_true)
        
        # Hamming Score: Fraction of correct labels total
        total_labels = len(multilabel_rows_true) * 3
        correct_labels = sum([sum([1 for j in range(3) if multilabel_rows_true[i][j] == multilabel_rows_pred[i][j]]) for i in range(len(multilabel_rows_true))])
        hamming_score = correct_labels / total_labels
        
        multilabel_metrics = {
            "Exact Match Ratio (All 3 Correct)": exact_match_ratio,
            "Hamming Score (Avg Accuracy)": hamming_score,
            "Fully Verified Samples": len(multilabel_rows_true)
        }

    return {
        "summary": summary_df,
        "confusion_matrices": confusion_matrices,
        "class_reports": class_reports,
        "multilabel": multilabel_metrics
    }

@st.fragment(run_every=30)
def display_predictions():
    """Display and verify predictions with auto-refresh."""
    # Verification Logic
    if 'predictions' in st.session_state and st.session_state.predictions:
        # Update all predictions with results
        for i, pred in enumerate(st.session_state.predictions):
            # Only check if pending or wait message
            if any(pred.get(f"t+{k} Result") not in ["✅ Correct", "❌ Incorrect", "Neutral", "Error"] for k in [1, 5, 30]):
                updated_pred = check_prediction_result(pred, pred['Ticker'])
                st.session_state.predictions[i] = updated_pred

    # Display Logic
    st.subheader("Predictions")
    if 'predictions' in st.session_state and st.session_state.predictions:
        # Create a display dataframe (hide internal columns)
        predictions_df = pd.DataFrame(st.session_state.predictions)
        
        # Filter columns to show
        cols_to_show = [
            "Ticker", "Timestamp", "Last Candle Time", "Identified Candlestick Patterns",
            "t+1 Prediction", "t+5 Prediction", "t+30 Prediction",
            "t+1 Result", "t+5 Result", "t+30 Result"
        ]
        
        # Ensure columns exist (for old dummy data compatibility)
        available_cols = [c for c in cols_to_show if c in predictions_df.columns]
        display_df = predictions_df[available_cols]
        
        # Apply styling
        def highlight_results(val):
            if isinstance(val, str):
                if "✅" in val:
                    return 'background-color: #d4edda; color: #155724' # Green
                elif "❌" in val:
                    return 'background-color: #f8d7da; color: #721c24' # Red
                elif "⏳" in val:
                    return 'background-color: #fff3cd; color: #856404' # Yellow
            return ''

        # Apply style to Result columns
        result_cols = [c for c in available_cols if "Result" in c]
        styled_df = display_df.style.map(highlight_results, subset=result_cols)
        
        st.dataframe(styled_df, width="stretch", hide_index=True)
        
        # Calculate and display metrics
        metrics_results = calculate_metrics(st.session_state.predictions)
        
        if not metrics_results["summary"].empty:
            st.subheader("Model Performance Metrics")
            
            # 1. Multi-label Metrics (Top Level)
            if metrics_results["multilabel"]:
                m = metrics_results["multilabel"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Exact Match Ratio (All 3)", f"{m['Exact Match Ratio (All 3 Correct)']:.1%}", help="Percentage of predictions where t+1, t+5, AND t+30 were ALL correct.")
                c2.metric("Hamming Score", f"{m['Hamming Score (Avg Accuracy)']:.1%}", help="Average accuracy across all horizons (e.g. 2/3 correct = 66%).")
                c3.metric("Fully Verified Samples", m['Fully Verified Samples'])
            
            # 2. Detailed Metrics Tabs
            tab_summary, tab_t1, tab_t5, tab_t30 = st.tabs(["Summary", "t+1 Metrics", "t+5 Metrics", "t+30 Metrics"])
            
            with tab_summary:
                st.caption("High-level performance overview.")
                format_dict = {"Accuracy": "{:.2%}", "F1 (Weighted)": "{:.2%}"}
                st.dataframe(metrics_results["summary"].style.format(format_dict), width="stretch", hide_index=True)
                
            # Helper to display detailed metrics for a horizon
            def display_horizon_metrics(k):
                if k in metrics_results["confusion_matrices"]:
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        st.markdown(f"**Confusion Matrix (t+{k})**")
                        st.dataframe(metrics_results["confusion_matrices"][k], width="stretch")
                    
                    with c2:
                        st.markdown(f"**Detailed Class Report (t+{k})**")
                        report = metrics_results["class_reports"][k]
                        # Format percentages
                        st.dataframe(report.style.format("{:.2%}"), width="stretch")
                else:
                    st.info(f"No verified data for t+{k} yet.")

            with tab_t1:
                display_horizon_metrics(1)
            with tab_t5:
                display_horizon_metrics(5)
            with tab_t30:
                display_horizon_metrics(30)
    else:
        st.info("No predictions yet. Click Submit or Refresh to generate predictions.")

# Streamlit app details
st.set_page_config(page_title="Financial Analysis", layout="wide")

# Initialize session state for ticker
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'AAPL'

# Initialize session state for predictions table
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

def get_last_30_min_data(stock):
    """
    Fetch the last 30 minutes of data with 1-minute intervals.
    If market is open, get the last 30 minutes from today.
    If market is closed, get the last 30 minutes from the most recent trading day.
    Returns up to 30 candles (or less if not enough data available).
    """
    # Try to get recent data with 1-minute intervals, INCLUDING extended hours
    # First, try fetching 1 day of 1-minute data
    history_1d = stock.history(period="1d", interval="1m", prepost=True)
    
    if not history_1d.empty and len(history_1d) > 0:
        # Take the last 30 candles (or all if less than 30)
        last_30 = history_1d.tail(30)
        return last_30
    
    # If no data from 1 day, try with 5 days
    history_5d = stock.history(period="5d", interval="1m", prepost=True)
    
    if not history_5d.empty and len(history_5d) > 0:
        # Take the last 30 candles (or all if less than 30)
        last_30 = history_5d.tail(30)
        return last_30
    
    # If still no data, return empty DataFrame
    return pd.DataFrame()

def plot_candlestick(data, ticker):
    """Plot candlestick chart using Plotly."""
    try:
        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=f'Stock Price - Last 30 Minutes',
            xaxis_rangeslider_visible=False,  # Hide default range slider for cleaner look
            height=600,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating candlestick chart: {e}")
        return None

# Top 30 US stocks by volume (approximate list for "Feeling Lucky")
TOP_30_STOCKS = [
    "NVDA", "TSLA", "AAPL", "AMD", "AMZN", "MSFT", "META", "GOOGL", "GOOG", "INTC",
    "COIN", "MARA", "PLTR", "SOFI", "BAC", "F", "T", "KVUE", "PFE", "VALE",
    "AAL", "CCL", "NCLH", "UBER", "LYFT", "SNAP", "RIVN", "LCID", "NIO", "XPEV"
]

def process_ticker(ticker, render_chart=True):
    """
    Process a single ticker: fetch data, predict, and optionally render chart.
    Returns True if successful, False otherwise.
    """
    try:
        # Retrieve stock data
        stock = yf.Ticker(ticker)
        
        # Get last 30 minutes of data with 1-minute intervals
        history = get_last_30_min_data(stock)
        
        if not history.empty:
            # Display candlestick chart if requested
            if render_chart:
                info = stock.info
                st.subheader(f"{ticker} - {info.get('longName', 'N/A')}")
                
                fig = plot_candlestick(history, ticker)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show data info
                st.caption(f"Showing {len(history)} candles from {history.index[0]} to {history.index[-1]}")

            # Add a new prediction row
            # Load model
            model, cdl_labels = load_model()
            
            if model:
                # Generate image for prediction
                img = generate_prediction_image(history)
                
                # Preprocess image for model
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension
                
                # Run prediction
                predictions = model.predict(img_batch, verbose=0) # verbose=0 to avoid stdout spam
                cdl_pred, price_pred = predictions
                
                # Process CDL patterns
                # Get top 3 patterns by probability
                top_k = 3
                top_indices = np.argsort(cdl_pred[0])[-top_k:][::-1]
                
                detected_patterns = []
                for i in top_indices:
                    prob = cdl_pred[0][i]
                    # Only show if probability is somewhat significant (e.g. > 0.1) to avoid noise
                    if prob > 0.1:
                        detected_patterns.append(f"{cdl_labels[i]} ({prob:.2f})")
                
                if detected_patterns:
                    patterns_str = ", ".join(detected_patterns)
                else:
                    patterns_str = "None (Highest: {:.2f})".format(np.max(cdl_pred[0]))
                    
                # Process Price Directions
                # Output: [next1_up, next1_down, next5_up, next5_down, next30_up, next30_down]
                def get_direction(up_prob, down_prob):
                    if up_prob > down_prob and up_prob > 0.5:
                        return "Up"
                    elif down_prob > up_prob and down_prob > 0.5:
                        return "Down"
                    else:
                        return "Neutral"
                        
                t1_pred = get_direction(price_pred[0][0], price_pred[0][1])
                t5_pred = get_direction(price_pred[0][2], price_pred[0][3])
                t30_pred = get_direction(price_pred[0][4], price_pred[0][5])
                
                # Get initial close and time for verification
                last_candle = history.iloc[-1]
                initial_close = float(last_candle['Close'])
                last_candle_time = history.index[-1]
                
                # Convert timestamps to Eastern Time for display
                et_tz = pytz.timezone('America/New_York')
                
                # Current time in ET
                timestamp_et = dt.now(et_tz).strftime("%Y-%m-%d %H:%M:%S ET")
                
                # Last candle time in ET
                if last_candle_time.tzinfo is None:
                    # If naive, assume UTC and convert (or assume ET if yfinance returned naive)
                    # yfinance usually returns aware. If naive, it's often market time (ET).
                    # Let's assume it's aware or we make it aware as UTC then convert.
                    # Safest is to assume it's already in correct time if naive, but let's try to handle aware.
                    last_candle_et_str = str(last_candle_time)
                else:
                    last_candle_et = last_candle_time.astimezone(et_tz)
                    last_candle_et_str = last_candle_et.strftime("%Y-%m-%d %H:%M:%S ET")
                
                # Create data for new prediction
                new_prediction = {
                    "Ticker": ticker,
                    "Timestamp": timestamp_et,
                    "Last Candle Time": last_candle_et_str,
                    "Initial Close": initial_close,
                    "Identified Candlestick Patterns": patterns_str,
                    "t+1 Prediction": t1_pred,
                    "t+5 Prediction": t5_pred,
                    "t+30 Prediction": t30_pred,
                    "t+1 Result": "Pending",
                    "t+5 Result": "Pending",
                    "t+30 Result": "Pending"
                }
            else:
                st.error("Model could not be loaded. Using dummy data.")
                # Fallback to dummy data if model fails
                new_prediction = {
                    "Ticker": ticker,
                    "Timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Last Candle Time": str(dt.now()), # Dummy
                    "Initial Close": 0.0, # Dummy
                    "Identified Candlestick Patterns": "Error",
                    "t+1 Prediction": "Error",
                    "t+5 Prediction": "Error",
                    "t+30 Prediction": "Error",
                    "t+1 Result": "Error",
                    "t+5 Result": "Error",
                    "t+30 Result": "Error"
                }
            
            # Add new prediction at the beginning of the list (newest first)
            st.session_state.predictions.insert(0, new_prediction)
            return True
        else:
            if render_chart:
                st.warning(f"No recent 1-minute interval data available for {ticker}.")
            return False
            
    except Exception as e:
        if render_chart:
            st.exception(f"An error occurred with {ticker}: {e}")
        print(f"Error processing {ticker}: {e}")
        return False

# Main app UI
st.title("Financial Analysis")

# Ticker input and buttons in the main area
col_ticker, col_submit, col_lucky = st.columns([3, 1, 1])
with col_ticker:
    ticker_input = st.text_input("Enter a stock ticker (e.g. AAPL)", value=st.session_state.ticker, label_visibility="collapsed", placeholder="Enter stock ticker (e.g. AAPL)")

with col_submit:
    submit = st.button("Submit", width="stretch")

with col_lucky:
    lucky = st.button("Feeling Lucky", width="stretch")

# Handle button clicks
# Handle ticker updates and button clicks
ticker_changed = False
clean_input = ticker_input.strip().upper()

# Check if input has changed (e.g. user typed new ticker and hit Enter, or clicked a button)
if clean_input and clean_input != st.session_state.ticker:
    st.session_state.ticker = clean_input
    ticker_changed = True

# 1. Auto-load AAPL on first run
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    st.session_state.first_run = False
    with st.spinner('Performing initial analysis for AAPL...', show_time=True):
        process_ticker('AAPL', render_chart=True)

# 2. Submit or Ticker Changed
if (submit or ticker_changed) and st.session_state.ticker:
    ticker = st.session_state.ticker
    if not ticker.strip():
        st.error("Please provide a valid stock ticker.")
    else:
        with st.spinner(f'Analyzing {ticker}...', show_time=True):
            process_ticker(ticker, render_chart=True)

# 3. Feeling Lucky (Batch Processing)
if lucky:
    # Use the first stock in the list for the chart
    first_ticker = TOP_30_STOCKS[0]
    st.session_state.ticker = first_ticker # Update session state ticker
    
    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_stocks = len(TOP_30_STOCKS)
    
    for i, ticker in enumerate(TOP_30_STOCKS):
        status_text.text(f"Processing {i+1}/{total_stocks}: {ticker}")
        
        # Render chart only for the first one
        should_render = (i == 0)
        
        process_ticker(ticker, render_chart=should_render)
        
        # Update progress
        progress_bar.progress((i + 1) / total_stocks)
    
    status_text.text("Batch processing complete!")
    time.sleep(1) # Let user see completion message
    status_text.empty()
    progress_bar.empty()

# Call the fragment always, outside the conditional blocks
display_predictions()

# Manual Verification Tool (Debug)
with st.expander("🛠️ Manual Verification (Debug)"):
    st.caption("Use this tool to check if data exists for a specific time.")
    c1, c2, c3 = st.columns(3)
    v_ticker = c1.text_input("Ticker", value="AAPL", key="v_ticker")
    v_date = c2.date_input("Date", value=dt.now(), key="v_date")
    v_time = c3.time_input("Time (Local)", value=dt.now().time(), key="v_time")
    
    if st.button("Check Data Availability"):
        try:
            # Construct target datetime (local)
            target_dt = datetime.combine(v_date, v_time)
            # Assume local timezone for input
            target_dt = target_dt.replace(tzinfo=dt.now().astimezone().tzinfo)
            
            st.write(f"Looking for data at: **{target_dt}** (Local) / **{target_dt.astimezone(datetime.timezone.utc)}** (UTC)")
            
            v_stock = yf.Ticker(v_ticker)
            # Fetch data with prepost=True
            v_df = v_stock.history(period="5d", interval="1m", prepost=True)
            
            if v_df.empty:
                st.error("No data found for the last 5 days.")
            else:
                # Convert df to UTC for comparison
                if v_df.index.tz is None:
                    v_df.index = v_df.index.tz_localize('UTC')
                else:
                    v_df.index = v_df.index.tz_convert('UTC')
                    
                target_utc = target_dt.astimezone(datetime.timezone.utc)
                
                # Find closest match
                closest_idx = None
                min_diff = float('inf')
                
                for idx in v_df.index:
                    diff = abs((idx - target_utc).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = idx
                        
                st.write(f"Closest data point found: **{closest_idx}** (UTC)")
                st.write(f"Difference: **{min_diff:.1f} seconds**")
                
                if min_diff < 60:
                    st.success(f"✅ Match found! Close: {v_df.loc[closest_idx]['Close']}")
                else:
                    st.warning("⚠️ No exact match found within 1 minute.")
                    
                st.dataframe(v_df.tail(5))
                
        except Exception as e:
            st.error(f"Error: {e}")