"""
Financial Time Series Forecasting with LSTM Neural Networks
Google Colab Compatible Version

This module provides comprehensive training and evaluation capabilities for LSTM-based
time series forecasting models optimized for Google Colab execution.

Author: Refactored for Originality
Version: 3.0.0
License: MIT
"""

# Google Colab optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Optimize for Colab
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Optimize for Colab

import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data_handler import DataPreprocessor
from utils.lstm_network import ForecastModel
from utils.baseline_methods import TraditionalForecasters


def plot_results(forecast_vals, actual_vals):
    """
    Create a visualization comparing forecasted and actual values.
    
    Generates a line plot showing the comparison between predicted and actual
    time series values. This visualization helps assess the quality of forecasts
    and identify patterns in prediction errors.
    
    Args:
        forecast_vals (numpy.ndarray): Array of forecasted values
        actual_vals (numpy.ndarray): Array of actual values
        
    Returns:
        matplotlib.figure.Figure: The created figure object
        
    Note:
        This function creates an interactive plot that can be displayed
        or saved for further analysis and reporting.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(actual_vals, label='Actual Data', linewidth=2)
    plt.plot(forecast_vals, label='LSTM Forecast', linewidth=2)
    plt.title('LSTM Time Series Forecasting Results', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_multiple_forecasts(forecast_vals, actual_vals, forecast_len):
    """
    Create a visualization for multiple forecast sequences.
    
    Generates a line plot showing multiple forecast sequences with proper
    alignment and padding. This is useful for visualizing multi-step
    forecasting results and comparing different prediction horizons.
    
    Args:
        forecast_vals (list): List of forecast sequences
        actual_vals (numpy.ndarray): Array of actual values
        forecast_len (int): Length of each forecast sequence
        
    Returns:
        matplotlib.figure.Figure: The created figure object
        
    Note:
        This function handles multiple forecast sequences by padding them
        appropriately to show their temporal alignment with actual data.
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(actual_vals, label='Actual Data', linewidth=2)
    
    # Plot each forecast sequence with proper padding
    for i, forecast_sequence in enumerate(forecast_vals):
        padding = [None for p in range(i * forecast_len)]
        plt.plot(padding + forecast_sequence, label=f'Forecast {i+1}', linewidth=1.5)
        
    plt.title('Multiple LSTM Forecast Sequences', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def calculate_metrics(forecast_vals, actual_vals):
    """
    Calculate comprehensive performance metrics for forecasting results.
    
    Computes various performance metrics including MSE, MAE, RMSE, and MAPE
    to evaluate the quality of forecasts. These metrics provide different
    perspectives on prediction accuracy and error characteristics.
    
    Args:
        forecast_vals (numpy.ndarray): Array of forecasted values
        actual_vals (numpy.ndarray): Array of actual values
        
    Returns:
        dict: Dictionary containing calculated performance metrics
        
    Note:
        These metrics provide comprehensive evaluation of forecast quality,
        with each metric emphasizing different aspects of prediction accuracy.
    """
    # Flatten arrays for metric calculation
    forecast_flat = forecast_vals.flatten()
    actual_flat = actual_vals.flatten()
    
    # Match lengths in case of mismatch
    min_len = min(len(forecast_flat), len(actual_flat))
    forecast_flat = forecast_flat[:min_len]
    actual_flat = actual_flat[:min_len]
    
    # Calculate various performance metrics
    model_mse = mean_squared_error(actual_flat, forecast_flat)
    model_mae = mean_absolute_error(actual_flat, forecast_flat)
    model_rmse = np.sqrt(model_mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_flat - forecast_flat) / actual_flat)) * 100
    
    # Calculate R-squared
    ss_res = np.sum((actual_flat - forecast_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    performance_metrics = {
        'mse': model_mse,
        'mae': model_mae,
        'rmse': model_rmse,
        'mape': mape,
        'r_squared': r_squared
    }
    
    print("\nLSTM Model Performance Metrics:")
    print("=" * 40)
    for metric, value in performance_metrics.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    return performance_metrics


def measure_trend_accuracy(forecast_vals, actual_vals):
    """
    Calculate directional accuracy for trend prediction.
    
    Measures how often the model correctly predicts the direction of change
    (up or down) compared to the actual values. This is particularly important
    for financial time series where trend direction matters more than exact values.
    
    Args:
        forecast_vals (numpy.ndarray): Array of forecasted values
        actual_vals (numpy.ndarray): Array of actual values
        
    Returns:
        float: Directional accuracy percentage
        
    Note:
        Directional accuracy is crucial for trading applications where
        predicting the correct direction of price movements is more important
        than predicting exact values.
    """
    # Calculate directional changes
    forecast_changes = np.diff(forecast_vals.flatten())
    actual_changes = np.diff(actual_vals.flatten())
    
    # Match lengths in case of mismatch
    min_len = min(len(forecast_changes), len(actual_changes))
    forecast_changes = forecast_changes[:min_len]
    actual_changes = actual_changes[:min_len]
    
    # Count correct directional predictions
    correct_directions = np.sum(np.sign(forecast_changes) == np.sign(actual_changes))
    total_predictions = len(forecast_changes)
    
    trend_accuracy = (correct_directions / total_predictions) * 100
    
    print(f"\nDirectional Accuracy: {trend_accuracy:.2f}%")
    return trend_accuracy


def compare_against_baselines(forecast_vals, actual_vals, train_data, test_data):
    """
    Compare LSTM performance against traditional forecasting methods.
    
    Evaluates the LSTM model's performance against traditional statistical
    forecasting methods including naive forecasting, moving averages, and ARIMA.
    This provides context for understanding the LSTM's relative performance.
    
    Args:
        forecast_vals (numpy.ndarray): LSTM forecasted values
        actual_vals (numpy.ndarray): Actual values
        train_data (numpy.ndarray): Training data for baseline methods
        test_data (numpy.ndarray): Testing data for baseline methods
        
    Returns:
        dict: Comparison results including all method performances
        
    Note:
        This comparison helps establish whether the LSTM model provides
        meaningful improvements over traditional forecasting approaches.
    """
    print("\nComparing LSTM against Traditional Methods...")
    print("=" * 50)
    
    # CRITICAL FIX: Use normalized data for fair comparison
    # The LSTM uses normalized data, so baselines must too
    # Extract the target variable (first column) from normalized data
    baseline_train = train_data[:, 0] if len(train_data.shape) > 1 else train_data
    baseline_test = actual_vals  # This is already the normalized test data
    
    # Ensure both arrays are 1D for baseline methods
    if len(baseline_train.shape) > 1:
        baseline_train = baseline_train.flatten()
    if len(baseline_test.shape) > 1:
        baseline_test = baseline_test.flatten()
    
    # Initialize traditional forecasters with normalized data
    traditional_forecasters = TraditionalForecasters(baseline_train, baseline_test)
    baseline_metrics = traditional_forecasters.run_all_methods()
    
    # Calculate LSTM metrics
    lstm_metrics = calculate_metrics(forecast_vals, actual_vals)
    
    # Add LSTM results to comparison
    baseline_metrics['lstm'] = {
        'predictions': forecast_vals,
        'mse': lstm_metrics['mse'],
        'mae': lstm_metrics['mae']
    }
    
    # Print comparison summary
    print("\nPerformance Comparison Summary:")
    print("=" * 50)
    for method, results in baseline_metrics.items():
        print(f"{method.upper()}:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print()
    
    return baseline_metrics


def main():
    """
    Main execution function for LSTM time series forecasting.
    
    Orchestrates the complete forecasting pipeline including data loading,
    model training, evaluation, and comparison with baseline methods.
    This function serves as the entry point for the forecasting application.
    
    Note:
        This function demonstrates the complete workflow for time series
        forecasting using LSTM neural networks with comprehensive evaluation.
    """
    print("🚀 Starting Financial Time Series Forecasting with LSTM")
    print("=" * 60)
    
    # Load configuration
    config_dict = json.load(open('model_config.json', 'r'))
    print(f"📊 Dataset: {config_dict['dataset']['csv_file']}")
    print(f"🔢 Epochs: {config_dict['training']['num_epochs']}")
    print(f"📈 Features: {config_dict['dataset']['features']}")
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(config_dict['architecture']['checkpoint_folder']):
        os.makedirs(config_dict['architecture']['checkpoint_folder'])

    # Initialize data processor
    print("\n📁 Loading and preprocessing data...")
    data_processor = DataPreprocessor(
        os.path.join('data', config_dict['dataset']['csv_file']),
        config_dict['dataset']['split_percentage'],
        config_dict['dataset']['features']
    )
    print(f"✅ Data loaded: {data_processor.train_size} train, {data_processor.test_size} test samples")

    # Initialize forecast model
    print("\n🧠 Building LSTM neural network...")
    forecast_model = ForecastModel()
    forecast_model.construct_network(config_dict)
    print("✅ LSTM network constructed successfully")
    
    # Prepare training data
    print("\n🔄 Preparing training sequences...")
    train_inputs, train_targets = data_processor.prepare_train_sequences(
        seq_len=config_dict['dataset']['lookback_period'],
        should_normalize=config_dict['dataset']['apply_normalization']
    )
    print(f"✅ Training sequences: {train_inputs.shape}")

    # Train the model using generator for memory efficiency
    print("\n🏋️ Training LSTM model...")
    batches_per_epoch = math.ceil((data_processor.train_size - config_dict['dataset']['lookback_period']) / 
                                 config_dict['training']['samples_per_batch'])
    
    forecast_model.train_with_generator(
        batch_generator=data_processor.create_train_batches(
            seq_len=config_dict['dataset']['lookback_period'],
            batch_count=config_dict['training']['samples_per_batch'],
            should_normalize=config_dict['dataset']['apply_normalization']
        ),
        num_epochs=config_dict['training']['num_epochs'],
        batch_count=config_dict['training']['samples_per_batch'],
        batches_per_epoch=batches_per_epoch,
        checkpoint_dir=config_dict['architecture']['checkpoint_folder']
    )

    # Prepare test data
    print("\n📊 Preparing test sequences...")
    test_inputs, test_targets = data_processor.prepare_test_sequences(
        seq_len=config_dict['dataset']['lookback_period'],
        should_normalize=config_dict['dataset']['apply_normalization']
    )
    print(f"✅ Test sequences: {test_inputs.shape}")

    # Generate forecasts using multiple sequence prediction
    print("\n🔮 Generating forecasts...")
    forecasts = forecast_model.forecast_multi_sequence(
        test_inputs, 
        config_dict['dataset']['lookback_period'], 
        config_dict['dataset']['lookback_period']
    )
    print(f"✅ Generated {len(forecasts)} forecast sequences")

    # Visualize results
    print("\n📈 Creating visualizations...")
    plot_multiple_forecasts(forecasts, test_targets, config_dict['dataset']['lookback_period'])

    # Calculate performance metrics
    print("\n📊 Calculating performance metrics...")
    performance_metrics = calculate_metrics(np.array(forecasts).flatten(), test_targets.flatten())
    
    # Measure directional accuracy
    trend_accuracy = measure_trend_accuracy(np.array(forecasts).flatten(), test_targets.flatten())
    
    # Compare against baseline methods
    # CRITICAL: Pass normalized data for fair comparison
    baseline_comparison = compare_against_baselines(
        np.array(forecasts).flatten(), 
        test_targets.flatten(),
        train_targets,  # Use normalized training targets
        test_targets    # Use normalized test targets
    )
    
    print("\n🎉 Forecasting pipeline completed successfully!")
    print("=" * 60)
    print(f"📊 Final LSTM MSE: {performance_metrics['mse']:.6f}")
    print(f"📊 Final LSTM MAE: {performance_metrics['mae']:.6f}")
    print(f"📈 Directional Accuracy: {trend_accuracy:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()