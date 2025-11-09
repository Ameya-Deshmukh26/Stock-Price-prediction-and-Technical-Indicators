"""
LSTM Implementation for Stock Price Prediction
Ready-to-use code that can be added to the Monte Carlo.ipynb notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("✓ TensorFlow imported successfully")
except ImportError:
    print("✗ TensorFlow not installed. Install with: pip install tensorflow")
    raise

def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM input
    
    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Number of past time steps to use for prediction
    
    Returns:
    --------
    X : numpy array
        Input sequences (samples, time_steps, features)
    y : numpy array
        Target values
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # Past values
        y.append(data[i])                # Next value
    return np.array(X), np.array(y)

def build_lstm_model(seq_length=60, lstm_units=[50, 50, 50], dropout_rate=0.2):
    """
    Build LSTM model architecture
    
    Parameters:
    -----------
    seq_length : int
        Length of input sequences
    lstm_units : list
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    model : Keras model
        Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(lstm_units[0], return_sequences=True, 
                   input_shape=(seq_length, 1)))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for units in lstm_units[1:]:
        model.add(LSTM(units, return_sequences=(units != lstm_units[-1])))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def forecast_future(model, last_sequence, steps=4, seq_length=60):
    """
    Forecast future values using rolling window approach
    
    Parameters:
    -----------
    model : Keras model
        Trained LSTM model
    last_sequence : array
        Last sequence from training data
    steps : int
        Number of future steps to forecast
    seq_length : int
        Length of input sequences
    
    Returns:
    --------
    forecasts : numpy array
        Forecasted values
    """
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape for model input
        input_seq = current_sequence.reshape(1, seq_length, 1)
        
        # Predict next value
        next_pred = model.predict(input_seq, verbose=0)
        forecasts.append(next_pred[0, 0])
        
        # Update sequence: remove first element, add prediction
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    return np.array(forecasts)

def implement_lstm(train, test, seq_length=60, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=1):
    """
    Complete LSTM implementation pipeline
    
    Parameters:
    -----------
    train : pandas Series
        Training data
    test : pandas Series
        Test data (for evaluation)
    seq_length : int
        Number of past time steps to use
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    validation_split : float
        Fraction of training data for validation
    verbose : int
        Verbosity level
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, metrics, and scaler
    """
    print("=" * 60)
    print("LSTM Model Implementation")
    print("=" * 60)
    
    # Step 1: Scale data
    print("\n1. Scaling data...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    print(f"   ✓ Data scaled to [0, 1] range")
    
    # Step 2: Create sequences
    print(f"\n2. Creating sequences (seq_length={seq_length})...")
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"   ✓ Training sequences: {X_train.shape}")
    print(f"   ✓ Test sequences: {X_test.shape if 'test' in locals() else 'N/A'}")
    
    # Step 3: Build model
    print("\n3. Building LSTM model...")
    model = build_lstm_model(seq_length=seq_length)
    print("   ✓ Model architecture:")
    model.summary()
    
    # Step 4: Train model
    print(f"\n4. Training model (epochs={epochs})...")
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
        shuffle=False  # Important: don't shuffle time series data
    )
    
    print("   ✓ Training completed")
    
    # Step 5: Make predictions on test set
    print("\n5. Making predictions...")
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform to original scale
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Step 6: Forecast future values
    print("6. Forecasting future values...")
    last_sequence = train_scaled[-seq_length:]
    future_forecast = forecast_future(model, last_sequence, steps=len(test), seq_length=seq_length)
    future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1)).flatten()
    
    # Step 7: Calculate metrics
    print("\n7. Calculating evaluation metrics...")
    
    # Test set metrics
    lstm_mae = mean_absolute_error(y_test_actual, test_predictions)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    lstm_mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    
    # Forecast metrics (comparing future forecast to actual test values)
    forecast_mae = mean_absolute_error(test.values, future_forecast)
    forecast_rmse = np.sqrt(mean_squared_error(test.values, future_forecast))
    forecast_mape = np.mean(np.abs((test.values - future_forecast) / test.values)) * 100
    
    print(f"\n   Test Set Metrics:")
    print(f"   MAE:  {lstm_mae:.2f}")
    print(f"   RMSE: {lstm_rmse:.2f}")
    print(f"   MAPE: {lstm_mape:.2f}%")
    
    print(f"\n   Forecast Metrics (future predictions):")
    print(f"   MAE:  {forecast_mae:.2f}")
    print(f"   RMSE: {forecast_rmse:.2f}")
    print(f"   MAPE: {forecast_mape:.2f}%")
    
    # Step 8: Plot results
    print("\n8. Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Training history
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Training History', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test set predictions
    axes[0, 1].plot(y_test_actual, 'ko-', label='Actual', linewidth=2, markersize=8)
    axes[0, 1].plot(test_predictions, 'r--', label='LSTM Predictions', linewidth=2, marker='s')
    axes[0, 1].set_title('Test Set Predictions', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Stock Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Future forecast vs actual
    axes[1, 0].plot(test.index, test.values, 'ko-', label='Actual', linewidth=2, markersize=8)
    axes[1, 0].plot(test.index, future_forecast, 'b--', label='LSTM Forecast', linewidth=2, marker='^')
    axes[1, 0].set_title('Future Forecast vs Actual', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Stock Price')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Prediction errors
    test_errors = np.abs(y_test_actual.flatten() - test_predictions.flatten())
    forecast_errors = np.abs(test.values - future_forecast)
    axes[1, 1].bar(['Test Set', 'Forecast'], 
                   [np.mean(test_errors), np.mean(forecast_errors)],
                   color=['blue', 'green'], alpha=0.7)
    axes[1, 1].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    results = {
        'model': model,
        'scaler': scaler,
        'history': history,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'future_forecast': future_forecast,
        'metrics': {
            'test_mae': lstm_mae,
            'test_rmse': lstm_rmse,
            'test_mape': lstm_mape,
            'forecast_mae': forecast_mae,
            'forecast_rmse': forecast_rmse,
            'forecast_mape': forecast_mape
        }
    }
    
    print("\n" + "=" * 60)
    print("LSTM Implementation Complete!")
    print("=" * 60)
    
    return results

# Example usage (to be added to notebook):
"""
# After running the notebook cells up to cell 8 (train/test split)

# Install TensorFlow if needed
# %pip install tensorflow

# Import the function
from LSTM_Implementation import implement_lstm

# Run LSTM model
lstm_results = implement_lstm(
    train=train,
    test=test,
    seq_length=60,      # Use past 60 weeks
    epochs=50,          # Training epochs
    batch_size=32,      # Batch size
    validation_split=0.2,
    verbose=1
)

# Access results
lstm_forecast = lstm_results['future_forecast']
lstm_metrics = lstm_results['metrics']

# Compare with other models
print(f"\nLSTM Forecast: {lstm_forecast}")
print(f"LSTM MAE: {lstm_metrics['forecast_mae']:.2f}")
print(f"LSTM RMSE: {lstm_metrics['forecast_rmse']:.2f}")
print(f"LSTM MAPE: {lstm_metrics['forecast_mape']:.2f}%")
"""

