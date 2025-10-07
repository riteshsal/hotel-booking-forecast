import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hotel Booking Forecast (ML Model)", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Forecast using Machine Learning")
st.write("Upload your dataset and forecast **future monthly bookings** using a Machine Learning model!")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    if 'bookings' not in df.columns or 'arrival_date' not in df.columns:
        st.error("❌ Dataset must have columns named 'arrival_date' and 'bookings'.")
        st.stop()

    # Convert 'arrival_date' to datetime, handling potential errors
    try:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    except Exception as e:
        st.error(f"Error converting 'arrival_date' to datetime: {e}")
        st.stop()

    # Aggregate to monthly
    df_agg = df.set_index('arrival_date')
    monthly_df = df_agg.resample('ME').agg({'bookings': 'sum'}).reset_index() # Use 'ME' for Month End

    st.subheader("📈 Historical Monthly Bookings")
    fig_hist, ax_hist = plt.subplots(figsize=(10,4))
    ax_hist.plot(monthly_df['arrival_date'], monthly_df['bookings'], marker='o')
    ax_hist.set_title("Historical Monthly Bookings")
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Number of Bookings")
    st.pyplot(fig_hist)

    # Feature Engineering
    monthly_df['month'] = monthly_df['arrival_date'].dt.month
    monthly_df['year'] = monthly_df['arrival_date'].dt.year
    monthly_df['time_index'] = range(1, len(monthly_df) + 1)
    monthly_df['quarter'] = monthly_df['arrival_date'].dt.quarter

    # Prepare data for modeling
    X_app = monthly_df[['time_index', 'month', 'year', 'quarter']]
    y_app = monthly_df['bookings']

    # Model Training
    model_choice = st.selectbox("🤖 Choose Model", ["RandomForest", "LinearRegression"])
    model = RandomForestRegressor(n_estimators=100, random_state=42) if model_choice == "RandomForest" else LinearRegression() # Using n_estimators=100 as a reasonable default

    try:
        model.fit(X_app, y_app)
        st.success(f"{model_choice} model trained successfully on uploaded data!")

        # Model Evaluation (on training data as we don't have a separate test set in the app)
        y_pred_train = model.predict(X_app)
        mae_train = mean_absolute_error(y_app, y_pred_train)
        r2_train = r2_score(y_app, y_pred_train)
        st.markdown("### ✅ Model Performance (on historical data)")
        st.write(f"**MAE:** {mae_train:.2f}")
        st.write(f"**R² Score:** {r2_train:.3f}")

        # --- Forecast Future ---
        st.subheader("🔮 Forecast Future Months")
        n_months = st.slider("Forecast horizon (months):", 1, 24, 6)

        # Determine the start date for forecasting
        last_historical_date_app = monthly_df['arrival_date'].max()
        start_forecast_date_app = last_historical_date_app + pd.DateOffset(months=1)

        future_dates = pd.date_range(start=start_forecast_date_app, periods=n_months, freq='ME')
        future_df = pd.DataFrame(index=future_dates)

        # Create features for future dates
        last_time_index_app = X_app['time_index'].max()
        future_df['time_index'] = range(last_time_index_app + 1, last_time_index_app + 1 + n_months)
        future_df['month'] = future_df.index.month
        future_df['year'] = future_df.index.year
        future_df['quarter'] = future_df.index.quarter

        # Ensure column order is the same as training data
        future_df = future_df[X_app.columns]


        # Make the future forecast
        future_forecast = model.predict(future_df)

        # Convert forecast to integer and ensure no negative predictions
        future_forecast = np.maximum(0, future_forecast.round().astype(int))

        # --- Visualize the forecast ---
        fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
        # Plot historical data
        ax_forecast.plot(monthly_df['arrival_date'], monthly_df['bookings'], marker='o', label='Historical Actual Bookings')
        # Plot forecast
        ax_forecast.plot(future_df.index, future_forecast, marker='x', linestyle='--', color='red', label=f'Forecast ({n_months} months)')

        ax_forecast.set_title('Hotel Bookings Forecast')
        ax_forecast.set_xlabel('Month')
        ax_forecast.set_ylabel('Number of Bookings')
        ax_forecast.legend()
        ax_forecast.grid(True)
        st.pyplot(fig_forecast)

        # --- Display forecast data ---
        st.subheader('Forecasted Bookings Data')
        forecast_results_df = pd.DataFrame({'Forecasted Bookings': future_forecast}, index=future_df.index)
        st.dataframe(forecast_results_df)

        # Download button
        csv = forecast_results_df.to_csv().encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "hotel_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred during model training or forecasting: {e}")
        st.stop()

else:
    st.info("👆 Please upload a dataset to generate the forecast.")
