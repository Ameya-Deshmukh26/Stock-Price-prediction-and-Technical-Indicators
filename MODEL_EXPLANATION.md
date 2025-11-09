# Stock Price Prediction Models: Interview Explanation Guide

## Overview of the Notebook

This notebook implements **three different forecasting approaches** for stock price prediction using Amazon (AMZN) weekly data:

1. **Monte Carlo Simulation** - Probabilistic approach using Geometric Brownian Motion
2. **ARIMA Model** - Classical time series forecasting
3. **SARIMA Model** - ARIMA with seasonal components

The workflow follows these steps:
- Data fetching and preprocessing (weekly stock prices from 2020-2025)
- Train/test split (last 4 weeks as test set)
- Model implementation and parameter optimization
- Model evaluation and comparison using MAE, RMSE, and MAPE metrics

---

## 1. Monte Carlo Simulation Implementation

### What It Does:
Monte Carlo simulation generates **thousands of possible future price paths** by modeling stock prices as a random walk using Geometric Brownian Motion (GBM).

### Key Steps:

1. **Calculate Log Returns**: 
   ```python
   log_returns = np.log(1 + train.pct_change())
   ```
   - Converts price changes to log returns (normalizes the distribution)
   - Log returns are more stable for financial modeling

2. **Estimate Parameters**:
   - **Drift (μ)**: Mean of log returns
   - **Volatility (σ)**: Standard deviation of log returns, annualized
   - **Time step (dt)**: Time increment for simulation

3. **Geometric Brownian Motion Formula**:
   ```
   S(t+1) = S(t) × exp((μ - 0.5×σ²)×dt + σ×√dt×Z)
   ```
   Where:
   - `S(t)` = current stock price
   - `Z` = random number from standard normal distribution
   - The term `(μ - 0.5×σ²)` is the drift adjustment

4. **Simulation:
   - Runs 1000 simulations (paths)
   - Each path generates 4 future price points
   - Final prediction = mean of all 1000 paths
   - Provides confidence intervals (standard deviation)

### Interview Answer:
*"Monte Carlo simulation models stock prices as a stochastic process following Geometric Brownian Motion. We estimate drift and volatility from historical returns, then simulate thousands of random price paths. The final forecast is the average of all paths, giving us both a point estimate and uncertainty bounds. This is particularly useful for risk analysis and understanding the distribution of possible outcomes."*

---

## 2. ARIMA Model Implementation

### What is ARIMA?

**ARIMA(p, d, q)** stands for:
- **AR (p)**: Autoregressive - uses `p` past values
- **I (d)**: Integrated - applies differencing `d` times to make series stationary
- **MA (q)**: Moving Average - uses `q` past forecast errors

### Implementation Steps:

1. **Stationarity Check**:
   ```python
   adfuller(timeseries)  # Augmented Dickey-Fuller test
   ```
   - Tests if the series has a unit root (non-stationary)
   - If p-value > 0.05, series is non-stationary → needs differencing

2. **Differencing**:
   ```python
   train_diff = train.diff().dropna()
   ```
   - Removes trend by taking first differences
   - Makes series stationary (constant mean and variance)

3. **Parameter Selection (p, d, q)**:
   - **ACF Plot**: Helps identify `q` (MA order) - where ACF cuts off
   - **PACF Plot**: Helps identify `p` (AR order) - where PACF cuts off
   - **Grid Search**: Tests combinations of (p,d,q) and selects best AIC (Akaike Information Criterion)

4. **Model Fitting**:
   ```python
   model = ARIMA(train, order=(p, d, q))
   fitted_model = model.fit()
   forecast = fitted_model.forecast(steps=4)
   ```

### Mathematical Formulation:

ARIMA(p,d,q) model:
```
(1 - φ₁B - φ₂B² - ... - φₚBᵖ)(1-B)ᵈyₜ = (1 + θ₁B + θ₂B² + ... + θₑBᵑ)εₜ
```
Where:
- `B` = backshift operator (Byₜ = yₜ₋₁)
- `(1-B)ᵈ` = differencing operator
- `εₜ` = white noise error term

### Interview Answer:
*"ARIMA models capture linear dependencies in time series data. The 'I' component handles non-stationarity through differencing, while AR and MA components model autocorrelation patterns. We use ACF/PACF plots and grid search with AIC to select optimal parameters. ARIMA works well for data with clear trends but struggles with non-linear patterns or strong seasonality."*

---

## 3. SARIMA Model Implementation

### What is SARIMA?

**SARIMA(p,d,q)(P,D,Q,s)** extends ARIMA by adding **seasonal components**:
- **(p,d,q)**: Non-seasonal ARIMA parameters
- **(P,D,Q)**: Seasonal ARIMA parameters
- **s**: Seasonal period (52 for weekly data, 12 for monthly, 4 for quarterly)

### Implementation Steps:

1. **Seasonal Decomposition**:
   ```python
   seasonal_decompose(train, model='additive', period=52)
   ```
   - Separates series into: Trend + Seasonal + Residual
   - Helps identify if seasonality exists

2. **Parameter Selection**:
   - Grid search over both non-seasonal (p,d,q) and seasonal (P,D,Q) parameters
   - Uses AIC to select best combination
   - More computationally expensive than ARIMA

3. **Model Fitting**:
   ```python
   model = SARIMAX(train, 
                   order=(p, d, q),
                   seasonal_order=(P, D, Q, s))
   fitted_model = model.fit()
   ```

### When to Use SARIMA:
- Data shows repeating patterns (e.g., quarterly earnings, holiday effects)
- Weekly/monthly data with annual cycles
- When ARIMA residuals show seasonal patterns

### Interview Answer:
*"SARIMA extends ARIMA by modeling seasonal patterns. For weekly stock data, we use s=52 to capture annual cycles. The model has both non-seasonal (p,d,q) and seasonal (P,D,Q) components. We perform seasonal decomposition first to confirm seasonality exists, then use grid search to optimize all parameters. SARIMA is more complex but essential when data exhibits recurring patterns."*

---

## 4. LSTM Implementation (How to Add It)

### What is LSTM?

**Long Short-Term Memory (LSTM)** is a type of Recurrent Neural Network (RNN) designed to:
- Remember long-term dependencies
- Handle sequences of data
- Capture non-linear patterns that ARIMA/SARIMA miss

### Why LSTM for Stock Prediction?
- **Non-linear relationships**: Can learn complex patterns
- **Multiple features**: Can use volume, technical indicators, etc.
- **Sequence learning**: Understands temporal dependencies better than linear models

### Implementation Steps:

#### Step 1: Data Preparation
```python
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Scale data to [0,1] range (LSTMs work better with normalized data)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))
```

#### Step 2: Create Sequences
```python
def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM input
    seq_length: number of past time steps to use for prediction
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # Past 60 values
        y.append(data[i])                # Next value
    return np.array(X), np.array(y)

seq_length = 60  # Use past 60 weeks to predict next week
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for LSTM: (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
```

#### Step 3: Build LSTM Model
```python
def build_lstm_model(seq_length=60):
    model = Sequential([
        # First LSTM layer with return_sequences=True
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),  # Prevent overfitting
        
        # Second LSTM layer
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(50),
        Dropout(0.2),
        
        # Output layer (single value prediction)
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_lstm_model(seq_length)
model.summary()
```

#### Step 4: Train the Model
```python
# Training parameters
epochs = 50
batch_size = 32

# Train model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1,
    shuffle=False  # Don't shuffle time series data
)
```

#### Step 5: Make Predictions
```python
# Predict on test set
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform to original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Forecast future values (rolling forecast)
def forecast_future(model, last_sequence, steps=4):
    """
    Forecast future values using rolling window
    """
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)
        forecasts.append(next_pred[0, 0])
        
        # Update sequence: remove first, add prediction
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    return np.array(forecasts)

# Get last sequence from training data
last_sequence = train_scaled[-seq_length:]
future_forecast = forecast_future(model, last_sequence, steps=4)
future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1)).flatten()
```

#### Step 6: Evaluate Model
```python
# Calculate metrics
lstm_mae = mean_absolute_error(y_test_actual, test_predictions)
lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
lstm_mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100

print(f"LSTM MAE: {lstm_mae:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")
print(f"LSTM MAPE: {lstm_mape:.2f}%")
```

### Complete LSTM Code Block for Notebook:

```python
## 5. LSTM Model Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_split=0.2, verbose=1, shuffle=False)

# Forecast
last_sequence = train_scaled[-seq_length:]
forecasts = []
current_seq = last_sequence.copy()

for _ in range(4):
    next_pred = model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
    forecasts.append(next_pred[0, 0])
    current_seq = np.append(current_seq[1:], next_pred)

lstm_forecast = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
```

### Interview Answer for LSTM:
*"LSTM is a deep learning approach that excels at capturing long-term dependencies and non-linear patterns in sequential data. Unlike ARIMA which assumes linear relationships, LSTM can learn complex patterns from historical sequences. We use a sliding window approach - feeding past 60 weeks to predict the next week. The model has multiple LSTM layers with dropout for regularization. LSTMs are particularly powerful when you have large datasets and want to incorporate multiple features like volume, technical indicators, or external factors. However, they require more data, computational resources, and careful hyperparameter tuning compared to classical methods."*

---

## Model Comparison Summary

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **Monte Carlo** | Risk analysis, uncertainty quantification, scenario planning | Doesn't learn patterns, assumes GBM | Risk assessment, option pricing |
| **ARIMA** | Simple, interpretable, works with small data | Linear only, no seasonality | Trend-based forecasting |
| **SARIMA** | Handles seasonality, still interpretable | More complex, slower | Seasonal patterns |
| **LSTM** | Non-linear patterns, multiple features, powerful | Needs lots of data, black box, computationally expensive | Complex patterns, large datasets |

---

## Key Interview Points

1. **When to use each model?**
   - **Monte Carlo**: Risk analysis, understanding uncertainty
   - **ARIMA**: Simple trend-based forecasting
   - **SARIMA**: When seasonality is present
   - **LSTM**: Complex non-linear patterns, large datasets

2. **Stationarity**: Why is it important?
   - ARIMA/SARIMA require stationary data (constant mean/variance)
   - Achieved through differencing
   - LSTM can handle non-stationary data but benefits from normalization

3. **Evaluation Metrics**:
   - **MAE**: Average absolute error (robust to outliers)
   - **RMSE**: Penalizes large errors more (sensitive to outliers)
   - **MAPE**: Percentage error (scale-independent)

4. **Trade-offs**:
   - **Classical (ARIMA/SARIMA)**: Interpretable, fast, work with small data
   - **Deep Learning (LSTM)**: More powerful, but needs data and compute

---

## Additional Notes for Interview

- **Ensemble Methods**: Combine multiple models (e.g., average ARIMA + LSTM predictions)
- **Feature Engineering**: For LSTM, can add technical indicators (RSI, MACD, moving averages)
- **Hyperparameter Tuning**: Use validation sets and cross-validation
- **Overfitting**: Monitor train vs validation loss, use dropout/regularization
- **Real-world Considerations**: Market regime changes, external shocks, model retraining frequency

