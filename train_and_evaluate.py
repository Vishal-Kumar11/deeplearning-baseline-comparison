"""
Financial Time Series Forecasting: LSTM vs GRU vs Traditional Methods
Google Colab Compatible Version

This module provides comprehensive training and evaluation capabilities for
LSTM and GRU-based time series forecasting models, compared against traditional
statistical baselines.

Author: Refactored for Originality
Version: 4.0.0
License: MIT
"""

# Google Colab optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Optimize for Colab
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Optimize for Colab

import json
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data_handler import DataPreprocessor
from utils.lstm_network import ForecastModel
from utils.gru_network import GRUForecastModel
from utils.baseline_methods import TraditionalForecasters


def plot_multiple_forecasts(forecast_vals, actual_vals, forecast_len, title='Forecast Sequences'):
    """
    Create a visualization for multiple forecast sequences.

    Args:
        forecast_vals (list): List of forecast sequences
        actual_vals (numpy.ndarray): Array of actual values
        forecast_len (int): Length of each forecast sequence
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(actual_vals, label='Actual Data', linewidth=2)

    for i, forecast_sequence in enumerate(forecast_vals):
        padding = [None for _ in range(i * forecast_len)]
        plt.plot(padding + forecast_sequence, label=f'Forecast {i+1}', linewidth=1.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_model_comparison(lstm_forecasts, gru_forecasts, actual_vals, forecast_len):
    """
    Plot LSTM and GRU forecasts side by side against actual values.

    Args:
        lstm_forecasts (list): LSTM forecast sequences
        gru_forecasts (list): GRU forecast sequences
        actual_vals (numpy.ndarray): Array of actual values
        forecast_len (int): Length of each forecast sequence

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    for ax, forecasts, model_name in zip(axes, [lstm_forecasts, gru_forecasts], ['LSTM', 'GRU']):
        ax.plot(actual_vals, label='Actual Data', linewidth=2, color='steelblue')
        for i, seq in enumerate(forecasts):
            padding = [None] * (i * forecast_len)
            ax.plot(padding + seq, linewidth=1.5, alpha=0.7)
        ax.set_title(f'{model_name} Forecast Sequences', fontsize=13, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(['Actual Data'], fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Steps', fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig


def calculate_metrics(forecast_vals, actual_vals, model_name='Model'):
    """
    Calculate comprehensive performance metrics for forecasting results.

    Args:
        forecast_vals (numpy.ndarray): Array of forecasted values
        actual_vals (numpy.ndarray): Array of actual values
        model_name (str): Name of the model for display purposes

    Returns:
        dict: Dictionary containing calculated performance metrics
    """
    forecast_flat = forecast_vals.flatten()
    actual_flat = actual_vals.flatten()

    min_len = min(len(forecast_flat), len(actual_flat))
    forecast_flat = forecast_flat[:min_len]
    actual_flat = actual_flat[:min_len]

    model_mse = mean_squared_error(actual_flat, forecast_flat)
    model_mae = mean_absolute_error(actual_flat, forecast_flat)
    model_rmse = np.sqrt(model_mse)

    mape = np.mean(np.abs((actual_flat - forecast_flat) / actual_flat)) * 100

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

    print(f"\n{model_name} Performance Metrics:")
    print("=" * 40)
    for metric, value in performance_metrics.items():
        print(f"{metric.upper()}: {value:.6f}")

    return performance_metrics


def measure_trend_accuracy(forecast_vals, actual_vals, model_name='Model'):
    """
    Calculate directional accuracy for trend prediction.

    Args:
        forecast_vals (numpy.ndarray): Array of forecasted values
        actual_vals (numpy.ndarray): Array of actual values
        model_name (str): Name of the model for display purposes

    Returns:
        float: Directional accuracy percentage
    """
    forecast_changes = np.diff(forecast_vals.flatten())
    actual_changes = np.diff(actual_vals.flatten())

    min_len = min(len(forecast_changes), len(actual_changes))
    forecast_changes = forecast_changes[:min_len]
    actual_changes = actual_changes[:min_len]

    correct_directions = np.sum(np.sign(forecast_changes) == np.sign(actual_changes))
    trend_accuracy = (correct_directions / len(forecast_changes)) * 100

    print(f"{model_name} Directional Accuracy: {trend_accuracy:.2f}%")
    return trend_accuracy


def compare_all_methods(lstm_forecasts, gru_forecasts, actual_vals, train_targets):
    """
    Compare LSTM and GRU performance against traditional forecasting methods.

    Args:
        lstm_forecasts (numpy.ndarray): LSTM forecasted values
        gru_forecasts (numpy.ndarray): GRU forecasted values
        actual_vals (numpy.ndarray): Actual test values
        train_targets (numpy.ndarray): Normalized training targets for baselines

    Returns:
        dict: Comparison results for all methods
    """
    print("\nComparing All Methods...")
    print("=" * 50)

    baseline_train = train_targets[:, 0] if len(train_targets.shape) > 1 else train_targets
    baseline_test = actual_vals.flatten()

    traditional_forecasters = TraditionalForecasters(baseline_train, baseline_test)
    all_results = traditional_forecasters.run_all_methods()

    lstm_metrics = calculate_metrics(lstm_forecasts, actual_vals, model_name='LSTM')
    gru_metrics = calculate_metrics(gru_forecasts, actual_vals, model_name='GRU')

    all_results['lstm'] = {
        'predictions': lstm_forecasts,
        'mse': lstm_metrics['mse'],
        'mae': lstm_metrics['mae']
    }
    all_results['gru'] = {
        'predictions': gru_forecasts,
        'mse': gru_metrics['mse'],
        'mae': gru_metrics['mae']
    }

    print("\nFull Performance Comparison:")
    print("=" * 55)
    print(f"{'Method':<20} {'MSE':>12} {'MAE':>12}")
    print("-" * 55)
    for method, results in all_results.items():
        print(f"{method.upper():<20} {results['mse']:>12.6f} {results['mae']:>12.6f}")
    print("=" * 55)

    return all_results


def main():
    """
    Main execution function for LSTM vs GRU time series forecasting comparison.

    Trains both LSTM and GRU models on S&P 500 data, evaluates against
    traditional baselines, and prints a comprehensive comparison summary.
    """
    print("🚀 Starting Financial Time Series Forecasting: LSTM vs GRU")
    print("=" * 60)

    config_dict = json.load(open('model_config.json', 'r'))
    print(f"📊 Dataset: {config_dict['dataset']['csv_file']}")
    print(f"🔢 Epochs: {config_dict['training']['num_epochs']}")
    print(f"📈 Features: {config_dict['dataset']['features']}")

    checkpoint_folder = config_dict['architecture']['checkpoint_folder']
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # Load data once — shared by both models
    print("\n📁 Loading and preprocessing data...")
    data_processor = DataPreprocessor(
        os.path.join('data', config_dict['dataset']['csv_file']),
        config_dict['dataset']['split_percentage'],
        config_dict['dataset']['features']
    )
    print(f"✅ Data loaded: {data_processor.train_size} train, {data_processor.test_size} test samples")

    seq_len = config_dict['dataset']['lookback_period']
    should_normalize = config_dict['dataset']['apply_normalization']
    batch_size = config_dict['training']['samples_per_batch']
    num_epochs = config_dict['training']['num_epochs']
    batches_per_epoch = math.ceil((data_processor.train_size - seq_len) / batch_size)

    print("\n🔄 Preparing sequences...")
    train_inputs, train_targets = data_processor.prepare_train_sequences(seq_len, should_normalize)
    test_inputs, test_targets = data_processor.prepare_test_sequences(seq_len, should_normalize)
    print(f"✅ Train: {train_inputs.shape} | Test: {test_inputs.shape}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n🧠 Building LSTM neural network...")
    lstm_model = ForecastModel()
    lstm_model.construct_network(config_dict)

    print("\n🏋️ Training LSTM model...")
    lstm_model.train_with_generator(
        batch_generator=data_processor.create_train_batches(seq_len, batch_size, should_normalize),
        num_epochs=num_epochs,
        batch_count=batch_size,
        batches_per_epoch=batches_per_epoch,
        checkpoint_dir=checkpoint_folder
    )

    print("\n🔮 Generating LSTM forecasts...")
    lstm_forecasts = lstm_model.forecast_multi_sequence(test_inputs, seq_len, seq_len)
    print(f"✅ Generated {len(lstm_forecasts)} LSTM forecast sequences")

    # ── GRU ───────────────────────────────────────────────────────────────────
    print("\n🧠 Building GRU neural network...")
    gru_model = GRUForecastModel()
    gru_model.construct_network(config_dict)

    print("\n🏋️ Training GRU model...")
    gru_model.train_with_generator(
        batch_generator=data_processor.create_train_batches(seq_len, batch_size, should_normalize),
        num_epochs=num_epochs,
        batch_count=batch_size,
        batches_per_epoch=batches_per_epoch,
        checkpoint_dir=checkpoint_folder
    )

    print("\n🔮 Generating GRU forecasts...")
    gru_forecasts = gru_model.forecast_multi_sequence(test_inputs, seq_len, seq_len)
    print(f"✅ Generated {len(gru_forecasts)} GRU forecast sequences")

    # ── Visualize ─────────────────────────────────────────────────────────────
    print("\n📈 Creating visualizations...")
    plot_model_comparison(lstm_forecasts, gru_forecasts, test_targets, seq_len)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n📊 Calculating directional accuracy...")
    lstm_trend = measure_trend_accuracy(np.array(lstm_forecasts).flatten(), test_targets.flatten(), 'LSTM')
    gru_trend = measure_trend_accuracy(np.array(gru_forecasts).flatten(), test_targets.flatten(), 'GRU')

    all_results = compare_all_methods(
        np.array(lstm_forecasts).flatten(),
        np.array(gru_forecasts).flatten(),
        test_targets.flatten(),
        train_targets
    )

    print("\n🎉 Forecasting pipeline completed successfully!")
    print("=" * 60)
    print(f"📊 LSTM  — MSE: {all_results['lstm']['mse']:.6f} | MAE: {all_results['lstm']['mae']:.6f} | Dir. Acc: {lstm_trend:.2f}%")
    print(f"📊 GRU   — MSE: {all_results['gru']['mse']:.6f}  | MAE: {all_results['gru']['mae']:.6f} | Dir. Acc: {gru_trend:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
