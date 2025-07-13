import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Sales Forecasting Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 Sales Forecasting Tool")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Function to load data with error handling for different encodings
# Sidebar for header option
header_option = st.sidebar.radio("Does the first row contain column headers?", ("Yes", "No"))

@st.cache_data
def load_data(file, header_option, encoding_option):
    try:
        if header_option == "Yes":
            header_row = 0  # Use the first row as headers
        else:
            header_row = None  # No headers in the file

        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding=encoding_option, header=header_row)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, header=header_row)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # If there's no header, create a default header
        if header_row is None:
            df.columns = [f"Column {i}" for i in range(1, len(df.columns) + 1)]
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    return df

# Sidebar for selecting encoding
encoding_option = st.sidebar.selectbox("Select file encoding", ['utf-8', 'ISO-8859-1', 'ASCII'])

# File upload with a unique key
uploaded_file = st.sidebar.file_uploader(
    "Upload your sales dataset (CSV or Excel)", 
    type=['csv', 'xls', 'xlsx'],
    key="file_uploader_1"  # Unique key to avoid DuplicateWidgetID error
)

# Call the function with the selected options
if uploaded_file is not None:
    df = load_data(uploaded_file, header_option, encoding_option)
    if df is not None:
        st.subheader("📂 Data Preview")
        st.dataframe(df.head())

        # Descriptive Statistics
        st.subheader("📊 Descriptive Statistics")
        st.write("Summary statistics provide insights into the data distribution.")
        st.write(df.describe())

        # Check if there's a date column
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        # If no datetime columns are found, try converting potential date columns to datetime
        if not date_cols:
            with st.expander("⚙️ Select a Date Column (if applicable)"):
                all_columns = df.columns.tolist()
                possible_date_column = st.selectbox("Select a column that might contain dates:", all_columns)

                # Try to convert the selected column to datetime
                try:
                    df[possible_date_column] = pd.to_datetime(df[possible_date_column], errors='coerce')
                    date_cols = [possible_date_column]
                    st.success(f"Successfully parsed {possible_date_column} as a date column.")
                except Exception as e:
                    st.error(f"Error converting {possible_date_column} to a date column: {e}")
                    st.stop()
        
        if not date_cols:
            st.error("No date column found. Please ensure your dataset includes a date column for time series forecasting.")
            st.stop()

        # Select the date column
        date_column = st.selectbox("Select the Date column", date_cols)

        # Drop rows with missing date
        df = df.dropna(subset=[date_column])

        # Sort data by date
        df = df.sort_values(by=date_column)

        # Data Visualization
        st.subheader("📈 Data Visualization")
        st.write("Visualize trends, seasonality, and relationships in the data.")

        # Line chart for sales trend
        st.write("### Sales Trend Over Time")
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_column], df.iloc[:, 1], label='Sales')  # Assuming sales is the 2nd column
        plt.title("Sales Over Time")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Correlation heatmap (for numeric columns only)
        st.write("### Correlation Heatmap")
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            st.write("No numeric columns available for correlation analysis.")
        else:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Between Features")
            st.pyplot(plt)

        # Select target variable
        st.subheader("🔍 Feature and Target Selection")
        selected_columns = [col for col in df.columns if col != date_column]
        target_column = st.selectbox("Select the Target Variable (e.g., Sales)", selected_columns)

        # Select predictor variables
        predictor_columns = st.multiselect("Select Predictor Variables (Features)", selected_columns, default=[])

        # Time Series Decomposition
        st.subheader("📊 Time Series Decomposition")
        if st.checkbox("Perform Time Series Decomposition"):
            # Ensure the target column (Sales) is numeric
            try:
                y = pd.to_numeric(df[target_column], errors='coerce')  # Convert to numeric, force non-numeric to NaN
                y = y.dropna()  # Drop any rows where conversion failed
                
                if len(y) > 0:
                    decomposition = seasonal_decompose(y, model='additive', period=12)
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
                    decomposition.trend.plot(ax=ax1, title='Trend')
                    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
                    decomposition.resid.plot(ax=ax3, title='Residual')
                    st.pyplot(plt)
                else:
                    st.error("The target column does not contain enough numeric data for decomposition.")
            except Exception as e:
                st.error(f"Error in time series decomposition: {e}")

        st.write(f"Selected Date Column: {date_column}")
        st.write(f"Selected Target Column: {target_column}")
        st.write(f"Selected Predictor Columns: {predictor_columns}")

        # Feature Engineering - Date Features
        st.subheader("📅 Feature Creation: Date Features")
        if st.checkbox("Extract Date Features"):
            df['Year'] = df[date_column].dt.year
            df['Month'] = df[date_column].dt.month
            df['DayOfWeek'] = df[date_column].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend flag

            st.write("Date features created: Year, Month, DayOfWeek, IsWeekend.")
            st.dataframe(df[['Year', 'Month', 'DayOfWeek', 'IsWeekend']].head())
            
        # Create a new DataFrame with selected date, predictor, and target columns
st.subheader("💾 Creating New Data for Target and Predictors")

if st.checkbox("Create New Data for Target and Predictors"):
    # Ensure all selected columns are available
    if date_column and target_column and predictor_columns:
        # Create a new DataFrame with date, predictors, and target columns
        data_final = df[[date_column] + predictor_columns + [target_column]].copy()

        # Ensure the DataFrame is sorted by the date column
        data_final = data_final.sort_values(by=date_column)

        # Display the new DataFrame
        st.write("New Data with Date, Predictor, and Target Columns:")
        st.dataframe(data_final.head())

        # Optionally, display a CSV download link for the new DataFrame
        csv = data_final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download new data as CSV",
            data=csv,
            file_name='sales_forecasting_data.csv',
            mime='text/csv',
        )
    else:
        st.error("Please make sure you have selected a date column, target column, and predictor columns.")

        # Feature Creation - Lag Features and Rolling Features
        st.subheader("🔧 Feature Creation: Lag and Rolling Features")
        if st.checkbox("Create Lag Features"):
            lag = st.slider("Select the number of lag periods (months) for creating lag features:", 1, 12, 3)
            for i in range(1, lag + 1):
                df[f'lag_{i}'] = df[target_column].shift(i)
            st.write(f"Lag features created for {lag} periods.")
            st.dataframe(df[[f'lag_{i}' for i in range(1, lag + 1)]].head())

        if st.checkbox("Create Rolling Features"):
            window = st.slider("Select the rolling window size (months):", 1, 12, 3)
            df[f'rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            st.write(f"Rolling mean feature created with a window size of {window} months.")
            st.dataframe(df[[f'rolling_mean_{window}']].head())

        # Encoding Categorical Variables
        st.subheader("🔠 Encoding Categorical Variables")
        if st.checkbox("Encode Categorical Variables"):
            encoder_type = st.selectbox("Select Encoding Type", ["Label Encoding", "One-Hot Encoding"])
            if encoder_type == "Label Encoding":
                label_encoders = {}
                for col in predictor_columns:
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle any mixed types
                        label_encoders[col] = le  # Save the encoder for future use (e.g., inverse transform)
                        st.write(f"Applied LabelEncoder to column: {col}")
            elif encoder_type == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=predictor_columns)
                st.write("One-Hot Encoding applied to categorical columns.")
            st.dataframe(df.head())

        # Model selection
        st.subheader("🧠 Model Selection")
        model_options = ['ARIMA', 'Exponential Smoothing', 'Linear Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']
        selected_model = st.selectbox("Choose a Forecasting Model", model_options)

        # Forecast horizon selection
        forecast_options = {'3 months': 3, '6 months': 6, '1 year': 12}
        forecast_choice = st.selectbox("Select Forecast Horizon", list(forecast_options.keys()))
        forecast_period = forecast_options[forecast_choice]

        # Run forecasting
        if st.button("🔮 Run Forecast"):
            st.subheader("🚀 Forecasting in Progress...")

            # Preprocessing
            data = df.copy()

            # Ensure date column is datetime and set it as the index
            data[date_column] = pd.to_datetime(data[date_column])

            # Set date as index
            data.set_index(date_column, inplace=True)

            # Check for duplicate index (dates) and handle them
            if data.index.duplicated().any():
                st.write("Duplicate dates found in the dataset. Aggregating values for duplicate dates.")
                
                # Here you can decide how to handle duplicates (e.g., taking the mean, sum, etc.)
                data = data.groupby(data.index).mean()  # This aggregates the data for duplicate dates

            # Ensure the index is unique before reindexing
            if data.index.duplicated().any():
                st.error("Duplicates still present after aggregation. Please ensure the data is clean.")
            else:
                # Handle missing values by reindexing with the correct frequency
                freq = pd.infer_freq(data.index)
                if freq is None:
                    freq = 'MS'  # Default frequency: Month start
                data = data.asfreq(freq, fill_value=np.nan).fillna(method='ffill')

                st.write("Data successfully reindexed and forward-filled for missing values.")

            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')

            # Define the number of periods to forecast
            freq = pd.infer_freq(data.index)
            if freq is None:
                freq = 'MS'  # Month start as default frequency
                data = data.asfreq(freq, fill_value=np.nan).fillna(method='ffill')

            # Determine the number of periods based on forecast_period
            if 'month' in freq or 'MS' in freq:
                periods = forecast_period
            elif 'D' in freq:
                periods = forecast_period * 30  # Approximate
            else:
                periods = forecast_period * 30  # Default to monthly

            # Split data into training and testing if needed (here we use all data for training)

            # Depending on the model, handle differently
            if selected_model in ['ARIMA', 'Exponential Smoothing']:
                # Time series models
                y = data[target_column]

                if selected_model == 'ARIMA':
                    # Simple ARIMA with automatic parameters (for demonstration)
                    model = ARIMA(y, order=(5,1,0))  # You might want to make p,d,q configurable
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=periods)
                
                elif selected_model == 'Exponential Smoothing':
                    # Holt-Winters Exponential Smoothing
                    model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=periods)
                
                # Create forecast dataframe
                forecast_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq=freq)
                forecast_df = pd.DataFrame({target_column: forecast}, index=forecast_dates)
            
            else:
                # Machine Learning models
                # Feature Engineering: create lag features
                df_ml = data.copy()
                lag_features = 3  # Number of lag months to use
                for lag in range(1, lag_features + 1):
                    df_ml[f'lag_{lag}'] = df_ml[target_column].shift(lag)
                df_ml = df_ml.dropna()

                # Define features and target
                X = df_ml[predictor_columns + [f'lag_{lag}' for lag in range(1, lag_features + 1)]]
                y = df_ml[target_column]

                # Split into training and testing if needed
                # For simplicity, we train on all available data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Initialize and train the selected model
                if selected_model == 'Linear Regression':
                    model = LinearRegression()
                elif selected_model == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif selected_model == 'Gradient Boosting':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                elif selected_model == 'Neural Network':
                    model = Sequential()
                    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
                    model.add(Dense(32, activation='relu'))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                else:
                    st.error("Unsupported model selected.")
                    st.stop()

                if selected_model != 'Neural Network':
                    model.fit(X_train, y_train)
                    # Forecasting
                    last_features = X.iloc[-1].values.reshape(1, -1)
                    predictions = []
                    current_features = last_features.copy()

                    for _ in range(periods):
                        pred = model.predict(current_features)[0]
                        predictions.append(pred)
                        # Update lag features
                        current_features = np.roll(current_features, -1)
                        current_features[0, -lag_features] = pred  # Assuming lag features are at the end
                else:
                    # Neural Network requires more handling
                    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
                    # Forecasting
                    last_features = X.iloc[-1].values.reshape(1, -1)
                    predictions = []
                    current_features = last_features.copy()

                    for _ in range(periods):
                        pred = model.predict(current_features)[0][0]
                        predictions.append(pred)
                        # Update lag features
                        current_features = np.roll(current_features, -1)
                        current_features[0, -lag_features] = pred

                # Create forecast dates
                forecast_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq=freq)
                forecast_df = pd.DataFrame({target_column: predictions}, index=forecast_dates)

            # Combine actual data with forecast
            combined_df = pd.concat([data[[target_column]], forecast_df])

            # Visualization
            st.subheader("📊 Forecast Results")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[target_column], mode='lines', name='Actual Sales'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[target_column], mode='lines', name='Predicted Sales'))
            fig.update_layout(title='Actual vs Predicted Sales',
                              xaxis_title='Date',
                              yaxis_title='Sales',
                              legend=dict(x=0, y=1),
                              template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

            # Display forecasted values
            st.subheader("📈 Forecasted Sales")
            st.write(forecast_df)

            # Optional: Display model performance metrics if applicable
            if selected_model not in ['ARIMA', 'Exponential Smoothing']:
                # Evaluate on test set
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.subheader("📊 Model Performance Metrics")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

else:
    st.info("Awaiting for CSV or Excel file to be uploaded.")
