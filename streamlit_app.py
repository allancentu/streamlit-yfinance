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
            
        # Parse the stored time (which includes timezone info from yfinance usually)
        last_candle_time = pd.to_datetime(prediction['Last Candle Time'])
        initial_close = prediction['Initial Close']
        
        # Convert last_candle_time to local system timezone for consistency with datetime.now()
        # This ensures "Wait until..." messages match the user's wall clock
        if last_candle_time.tzinfo is not None:
            last_candle_time = last_candle_time.astimezone(None) # None = local timezone
        
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
            if prediction[result_key] in ["Correct", "Incorrect", "Neutral"]:
                continue
                
            target_time = last_candle_time + timedelta(minutes=minutes)
            
            # Current time (naive, local)
            now = dt.now()
            # If target_time is aware, make now aware (local) or make target_time naive (local)
            # Since we converted last_candle_time to local (astimezone(None)), it might still be aware.
            # Let's ensure we compare apples to apples.
            if target_time.tzinfo is not None:
                # target_time is aware (local), so we need aware now
                now = dt.now().astimezone()
            
            # Define wait times (when to allow checking)
            # We check 1 minute after the candle closes to ensure data availability
            wait_delays = {
                't+1': 2,   # 1 min candle + 1 min buffer
                't+5': 6,   # 5 min horizon + 1 min buffer
                't+30': 31  # 30 min horizon + 1 min buffer
            }
            
            check_time = last_candle_time + timedelta(minutes=wait_delays[horizon_name])
            
            # If current time is before the check time, tell user to wait
            if check_time > now:
                # Format check time for display
                wait_time_str = check_time.strftime("%H:%M")
                prediction[result_key] = f"⏳ Wait until {wait_time_str}"
                continue
            
            # Target time is in the past, try to fetch data
            try:
                # Fetch 1m data for the last day (more robust than specific start/end times)
                df = stock.history(period="1d", interval="1m")
                
                # If 1d is empty, try 5d (e.g. over weekend)
                if df.empty:
                    df = stock.history(period="5d", interval="1m")
                
                if df.empty:
                    # Instead of "Data Unavailable", wait 1 more minute and try again
                    next_retry = dt.now() + timedelta(minutes=1)
                    prediction[result_key] = f"⏳ Wait until {next_retry.strftime('%H:%M')}"
                    continue
                
                # Convert df index to local timezone to match target_time
                df.index = df.index.tz_convert(None) if target_time.tzinfo is None else df.index.tz_convert(target_time.tzinfo)
                
                # Find the candle at target_time (or very close to it)
                # We look for the exact minute
                target_candle = None
                
                # Iterate to find exact match
                for idx, row in df.iterrows():
                    # Compare down to the minute
                    if idx.replace(second=0, microsecond=0) == target_time.replace(second=0, microsecond=0):
                        target_candle = row
                        break
                
                if target_candle is None:
                    # If exact match not found, maybe it's a gap? Wait 1 more minute and try again
                    next_retry = dt.now() + timedelta(minutes=1)
                    prediction[result_key] = f"⏳ Wait until {next_retry.strftime('%H:%M')}"
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
            "Ticker", "Timestamp", "Identified Candlestick Patterns",
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
    # Try to get recent data with 1-minute intervals
    # First, try fetching 1 day of 1-minute data
    history_1d = stock.history(period="1d", interval="1m")
    
    if not history_1d.empty and len(history_1d) > 0:
        # Take the last 30 candles (or all if less than 30)
        last_30 = history_1d.tail(30)
        return last_30
    
    # If no data from 1 day, try with 5 days
    history_5d = stock.history(period="5d", interval="1m")
    
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

# Main app UI
st.title("Financial Analysis")

# Ticker input and buttons in the main area
col_ticker, col_submit, col_refresh = st.columns([3, 1, 1])
with col_ticker:
    ticker_input = st.text_input("Enter a stock ticker (e.g. AAPL)", value=st.session_state.ticker, label_visibility="collapsed", placeholder="Enter stock ticker (e.g. AAPL)")

with col_submit:
    submit = st.button("Submit", width="stretch")

with col_refresh:
    refresh = st.button("Refresh", width="stretch")

# Handle button clicks
# Handle ticker updates and button clicks
ticker_changed = False
clean_input = ticker_input.strip().upper()

# Check if input has changed (e.g. user typed new ticker and hit Enter, or clicked a button)
if clean_input and clean_input != st.session_state.ticker:
    st.session_state.ticker = clean_input
    ticker_changed = True

# Trigger analysis if Submit, Refresh, or Ticker Changed
if (submit or refresh or ticker_changed) and st.session_state.ticker:
    ticker = st.session_state.ticker
    
    if not ticker.strip():
        st.error("Please provide a valid stock ticker.")
    else:
        try:
            with st.spinner('Fetching data...', show_time=True):
                # Retrieve stock data
                stock = yf.Ticker(ticker)
                info = stock.info

                st.subheader(f"{ticker} - {info.get('longName', 'N/A')}")

                # Get last 30 minutes of data with 1-minute intervals
                history = get_last_30_min_data(stock)
                
                if not history.empty:
                    # Display candlestick chart
                    fig = plot_candlestick(history, ticker)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data info
                    st.caption(f"Showing {len(history)} candles from {history.index[0]} to {history.index[-1]}")

                    # Add a new prediction row
                    from datetime import datetime
                    
                    # Load model
                    model, cdl_labels = load_model()
                    
                    if model:
                        # Generate image for prediction
                        img = generate_prediction_image(history)
                        
                        # Preprocess image for model
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension
                        
                        # Run prediction
                        predictions = model.predict(img_batch)
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
                        
                        # Create data for new prediction
                        new_prediction = {
                            "Ticker": ticker,
                            "Timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Last Candle Time": str(last_candle_time),
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
                        import random
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
                else:
                    st.warning("No recent 1-minute interval data available for this ticker. The market may be closed or this ticker may not support 1-minute data.")
                
        except Exception as e:
            st.exception(f"An error occurred: {e}")

# Call the fragment always, outside the conditional blocks
display_predictions()