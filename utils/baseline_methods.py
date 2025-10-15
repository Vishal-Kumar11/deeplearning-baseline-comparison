"""
TraditionalForecasters: Traditional statistical forecasting methods for baseline comparison

This class implements various traditional statistical forecasting methods including
naive forecasting, moving averages, and ARIMA models. These methods serve as
baselines for comparing the performance of LSTM neural networks.

Attributes:
    train_series: Training time series data
    test_series: Testing time series data
    forecast_vals: Forecasted values from various methods
    rolling_window: Rolling window for moving average calculations
    forecast_val: Single forecasted value
    arima_model: ARIMA model instance
    trained_arima: Fitted ARIMA model
    method_results: Results from all forecasting methods
    last_val_forecast: Naive forecast results
    moving_avg_forecast: Moving average forecast results
    arima_forecast: ARIMA forecast results

Example:
    forecaster = TraditionalForecasters(train_data, test_data)
    results = forecaster.run_all_methods()
    naive_pred = forecaster.last_value_forecast()
"""

# Fix TensorFlow threading issues on macOS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class TraditionalForecasters:
    """
    A comprehensive class implementing traditional statistical forecasting methods.
    
    This class provides various baseline forecasting methods including naive forecasting,
    moving averages, and ARIMA models. These methods are essential for establishing
    performance baselines and comparing against more advanced neural network approaches.
    """

    def __init__(self, train_series, test_series):
        """
        Initialize the TraditionalForecasters with training and testing data.
        
        Sets up the forecaster with the provided time series data for training
        and testing various traditional forecasting methods.
        
        Args:
            train_series (numpy.ndarray): Training time series data (can be normalized)
            test_series (numpy.ndarray): Testing time series data (can be normalized)
            
        Raises:
            ValueError: If input data is empty or invalid
        """
        # Handle both 1D and 2D input arrays
        # If 2D, extract first column (the target variable)
        if len(train_series.shape) > 1:
            self.train_series = train_series[:, 0]
        else:
            self.train_series = train_series
            
        if len(test_series.shape) > 1:
            self.test_series = test_series[:, 0]
        else:
            self.test_series = test_series
            
        self.forecast_vals = None
        self.method_results = {}

    def last_value_forecast(self):
        """
        Implement naive forecasting using the last observed value.
        
        The naive forecast method simply uses the last observed value as the
        prediction for all future time steps. This serves as a simple baseline
        for comparison with more sophisticated methods.
        
        Returns:
            numpy.ndarray: Array of naive forecasts
            
        Note:
            This method assumes that the most recent value is the best predictor
            of future values, which is often reasonable for short-term forecasting.
        """
        # Use the last value from training data as the forecast
        last_value = self.train_series[-1]
        forecast_vals = np.full(len(self.test_series), last_value)
        
        # Calculate performance metrics
        mse = mean_squared_error(self.test_series, forecast_vals)
        mae = mean_absolute_error(self.test_series, forecast_vals)
        
        self.method_results['naive'] = {
            'predictions': forecast_vals,
            'mse': mse,
            'mae': mae
        }
        
        return forecast_vals

    def rolling_mean_forecast(self, rolling_window=10):
        """
        Implement moving average forecasting.
        
        The moving average method uses the average of the last 'rolling_window'
        observations as the prediction for the next time step. This method
        smooths out short-term fluctuations and captures trends.
        
        Args:
            rolling_window (int): Number of previous observations to average
            
        Returns:
            numpy.ndarray: Array of moving average forecasts
            
        Note:
            The rolling window size affects the smoothness of predictions.
            Smaller windows are more responsive to recent changes, while
            larger windows provide smoother predictions.
        """
        forecast_vals = []
        
        # Calculate moving average for each test point
        for i in range(len(self.test_series)):
            if i < rolling_window:
                # Use available data for initial predictions
                available_data = self.train_series[-(rolling_window-i):]
                available_data = np.concatenate([available_data, forecast_vals])
                forecast_val = np.mean(available_data[-rolling_window:])
            else:
                # Use previous predictions for moving average
                forecast_val = np.mean(forecast_vals[-rolling_window:])
                
            forecast_vals.append(forecast_val)
        
        forecast_vals = np.array(forecast_vals)
        
        # Calculate performance metrics
        mse = mean_squared_error(self.test_series, forecast_vals)
        mae = mean_absolute_error(self.test_series, forecast_vals)
        
        self.method_results['moving_average'] = {
            'predictions': forecast_vals,
            'mse': mse,
            'mae': mae
        }
        
        return forecast_vals

    def statistical_arima_forecast(self, order=(1, 1, 1)):
        """
        Implement ARIMA (AutoRegressive Integrated Moving Average) forecasting.
        
        ARIMA is a statistical method that combines autoregression, differencing,
        and moving averages to model time series data. It's particularly effective
        for data with trends and seasonal patterns.
        
        Args:
            order (tuple): ARIMA order (p, d, q) where:
                p: Number of autoregressive terms
                d: Number of differencing operations
                q: Number of moving average terms
                
        Returns:
            numpy.ndarray: Array of ARIMA forecasts
            
        Raises:
            ValueError: If ARIMA model fails to fit
            Warning: If model convergence issues occur
            
        Note:
            ARIMA models require careful parameter selection. The (1,1,1) order
            is a common starting point, but optimal parameters may vary by dataset.
        """
        try:
            # Prepare data for ARIMA (use 1D array directly)
            train_data = self.train_series
            
            # Fit ARIMA model
            arima_model = ARIMA(train_data, order=order)
            trained_arima = arima_model.fit()
            
            # Generate forecasts
            forecast_result = trained_arima.forecast(steps=len(self.test_series))
            forecast_vals = forecast_result.values if hasattr(forecast_result, 'values') else forecast_result
            
            # Calculate performance metrics
            mse = mean_squared_error(self.test_series, forecast_vals)
            mae = mean_absolute_error(self.test_series, forecast_vals)
            
            self.method_results['arima'] = {
                'predictions': forecast_vals,
                'mse': mse,
                'mae': mae,
                'model': trained_arima
            }
            
            return forecast_vals
            
        except Exception as e:
            print(f"ARIMA forecasting failed: {e}")
            # Return naive forecast as fallback
            return self.last_value_forecast()

    def run_all_methods(self):
        """
        Execute all traditional forecasting methods and return results.
        
        Runs all implemented forecasting methods and returns a comprehensive
        comparison of their performance. This is useful for establishing
        baselines and comparing against neural network approaches.
        
        Returns:
            dict: Dictionary containing results from all forecasting methods
            
        Note:
            This method provides a comprehensive evaluation of traditional
            forecasting approaches, making it easy to compare performance
            across different methods and against neural network models.
        """
        print("Running traditional forecasting methods...")
        
        # Run all forecasting methods
        naive_pred = self.last_value_forecast()
        moving_avg_pred = self.rolling_mean_forecast()
        arima_pred = self.statistical_arima_forecast()
        
        # Print performance summary
        print("\nTraditional Forecasting Methods Performance:")
        print("=" * 50)
        
        for method, results in self.method_results.items():
            print(f"{method.upper()}:")
            print(f"  MSE: {results['mse']:.6f}")
            print(f"  MAE: {results['mae']:.6f}")
            print()
        
        return self.method_results
