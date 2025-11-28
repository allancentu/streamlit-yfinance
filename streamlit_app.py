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
from PIL import Image
from datetime import datetime as dt, timedelta

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
        model.load_weights('best_model.weights.h5')
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
    submit = st.button("Submit", use_container_width=True)

with col_refresh:
    refresh = st.button("Refresh", use_container_width=True)

# Handle button clicks
if submit:
    st.session_state.ticker = ticker_input.strip().upper()

if refresh and st.session_state.ticker:
    # Refresh with current ticker
    pass

# If we have a ticker to display (either from submit or refresh)
if (submit or refresh) and st.session_state.ticker:
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
                else:
                    st.warning("No recent 1-minute interval data available for this ticker. The market may be closed or this ticker may not support 1-minute data.")

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
                    
                    # Process CDL patterns (threshold 0.5)
                    detected_patterns_indices = np.where(cdl_pred[0] > 0.5)[0]
                    if len(detected_patterns_indices) > 0:
                        detected_patterns = [cdl_labels[i] for i in detected_patterns_indices]
                        patterns_str = ", ".join(detected_patterns)
                    else:
                        patterns_str = "None"
                        
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
                    
                    # Create data for new prediction
                    new_prediction = {
                        "Ticker": ticker,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Identified Candlestick Patterns": patterns_str,
                        "t+1 Prediction": t1_pred,
                        "t+5 Prediction": t5_pred,
                        "t+30 Prediction": t30_pred,
                        "t+1 Result": "",
                        "t+5 Result": "",
                        "t+30 Result": ""
                    }
                else:
                    st.error("Model could not be loaded. Using dummy data.")
                    # Fallback to dummy data if model fails
                    import random
                    new_prediction = {
                        "Ticker": ticker,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Identified Candlestick Patterns": "Error",
                        "t+1 Prediction": "Error",
                        "t+5 Prediction": "Error",
                        "t+30 Prediction": "Error",
                        "t+1 Result": "",
                        "t+5 Result": "",
                        "t+30 Result": ""
                    }
                
                # Add new prediction at the beginning of the list (newest first)
                st.session_state.predictions.insert(0, new_prediction)
                
                # Display predictions table
                st.subheader("Predictions")
                if st.session_state.predictions:
                    predictions_df = pd.DataFrame(st.session_state.predictions)
                    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No predictions yet. Click Submit or Refresh to generate predictions.")

        except Exception as e:
            st.exception(f"An error occurred: {e}")

