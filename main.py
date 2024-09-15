import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import cmdstanpy
import os

# Install cmdstan binaries if not already installed
if not os.path.exists(cmdstanpy.cmdstan_path()):
    cmdstanpy.install_cmdstan()

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# User input for stock symbol
stock_input = st.text_input("Enter stock symbol (e.g., AAPL, TATAMOTORS.BO):", "AAPL")

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    # Download stock data using yfinance
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            return None  # Return None if the stock symbol is invalid
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

# Load the data for the given stock
with st.spinner('Loading data...'):
    data = load_data(stock_input)

if data is None:
    st.error(f"Error: Unable to load data for {stock_input}. Please check the stock symbol.")
else:
    st.success("Data loaded successfully!")

    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    # Call the function to plot the raw data
    plot_raw_data()

    # Forecasting with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Plot forecast
    st.write('Forecast Data')
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)

    # Plot forecast components
    st.write('Forecast Component')
    fig1 = m.plot_components(forecast)
    st.write(fig1)
