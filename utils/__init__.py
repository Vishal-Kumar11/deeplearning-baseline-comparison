"""
Utils Package: Comprehensive utilities for time series forecasting

This package provides essential utilities for LSTM-based time series forecasting,
including data preprocessing, model construction, timing utilities, and baseline
forecasting methods for performance comparison.

Modules:
    data_handler: Data preprocessing and sequence generation
    lstm_network: LSTM neural network model implementation
    timer: Performance timing utilities
    baseline_methods: Traditional forecasting methods for comparison

Example:
    from utils.data_handler import DataPreprocessor
    from utils.lstm_network import ForecastModel
    from utils.timer import TimeTracker
    from utils.baseline_methods import TraditionalForecasters
"""

__version__ = "3.0.0"
__author__ = "Refactored for Originality"
__license__ = "MIT"

# Import main classes for easy access
from .data_handler import DataPreprocessor
from .lstm_network import ForecastModel
from .timer import TimeTracker
from .baseline_methods import TraditionalForecasters

__all__ = [
    'DataPreprocessor',
    'ForecastModel', 
    'TimeTracker',
    'TraditionalForecasters'
]
