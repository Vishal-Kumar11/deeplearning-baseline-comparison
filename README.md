# Deep Learning Baseline Comparison

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A systematic comparison of deep learning (LSTM) vs traditional statistical methods for time series forecasting.

---

GitHub Repository: `deeplearning-baseline-comparison`

# Financial Time Series Forecasting using LSTM Neural Networks

A comprehensive machine learning system for predicting financial time series data using LSTM neural networks and traditional statistical methods. This project implements a complete forecasting pipeline with model comparison and performance evaluation.

## Overview

This project implements an end-to-end time series forecasting pipeline using LSTM neural networks on S&P 500 data. The system compares deep learning approaches against traditional statistical methods and provides comprehensive evaluation metrics for financial forecasting applications.

## Features

- **LSTM Neural Network**: 3-layer stacked LSTM architecture with dropout regularization
- **Traditional Baselines**: Naive forecasting, Moving Average, and ARIMA models
- **Comprehensive Evaluation**: MSE, MAE, RMSE, R², and Directional Accuracy metrics
- **Memory-Efficient Processing**: Batch generation for large datasets
- **Configurable Architecture**: JSON-based model configuration
- **Professional Documentation**: Comprehensive docstrings and comments
- **Google Colab Compatible**: Optimized for cloud-based execution

## Dataset

The S&P 500 Dataset contains historical stock market data with multiple features:
- Close Price, Volume
- 25 years of trading data (2000-2025)
- 6,485 trading days
- Configurable train/test split (85% training, 15% testing)

## Technical Stack

- **Backend**: Python 3.10+, TensorFlow 2.15+
- **ML Libraries**: Keras, scikit-learn, statsmodels
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Statistical Analysis**: ARIMA, Moving Averages
- **Platform**: Google Colab Compatible

## Installation

### Google Colab Setup

1. Upload the project files to Google Colab
2. Install dependencies:
```python
!pip install tensorflow>=2.15.0
!pip install statsmodels>=0.14.0
!pip install scikit-learn>=1.3.0
!pip install matplotlib>=3.7.0
```

3. Run the forecasting pipeline:
```python
!python train_and_evaluate.py
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/deeplearning-baseline-comparison.git
cd deeplearning-baseline-comparison
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python train_and_evaluate.py
```

## Usage

### Basic Usage

1. **Configure your model** (optional)
Edit `model_config.json` to modify:
- Sequence length (lookback period)
- Number of epochs
- Batch size
- Layer architecture

2. **Run training and evaluation**
```bash
python train_and_evaluate.py
```

3. **View results**
- Console output shows comparative metrics
- Visualization plot displays predictions vs actual
- Model checkpoints saved in `saved_models/`

### Google Colab Usage

```python
# Upload your data files
from google.colab import files
uploaded = files.upload()

# Run the forecasting pipeline
!python train_and_evaluate.py

# View results
import matplotlib.pyplot as plt
plt.show()
```

## Project Structure

```
├── utils/                      # Core utilities package
│   ├── __init__.py
│   ├── data_handler.py        # Data preprocessing and sequence generation
│   ├── lstm_network.py        # LSTM model implementation
│   ├── timer.py               # Performance timing utilities
│   └── baseline_methods.py    # Traditional forecasting methods
├── data/
│   ├── sp500.csv              # S&P 500 historical data
│   └── sinewave.csv           # Test data
├── saved_models/              # Model checkpoints (created on run)
├── train_and_evaluate.py      # Main training and evaluation script
├── model_config.json          # Model configuration
├── requirements.txt           # Dependencies
└── README.md
```

## Configuration

The `model_config.json` file controls all model parameters:

```json
{
    "dataset": {
        "csv_file": "sp500.csv",
        "features": ["Close", "Volume"],
        "lookback_period": 50,
        "split_percentage": 0.85,
        "apply_normalization": true
    },
    "training": {
        "num_epochs": 50,
        "samples_per_batch": 32
    },
    "architecture": {
        "loss_function": "mse",
        "optimization_method": "adam",
        "checkpoint_folder": "saved_models",
        "layers": [...]
    }
}
```

## 📊 Results & Evaluation

Performance evaluation on S&P 500 data (6,485 trading days, 2000-2025):

### Comparative Analysis

| Method | MSE | MAE | Training Time | Improvement |
|--------|-----|-----|---------------|-------------|
| Naive Forecast | 0.005232 | 0.053960 | <1s | Baseline |
| Moving Average | 0.005060 | 0.053165 | <1s | 3.3% |
| ARIMA | 0.005212 | 0.053868 | ~2min | 0.4% |
| **LSTM** | **0.004496** | **0.048416** | **~6min** | **14.1%** |

### Key Findings

- **LSTM demonstrates 11-14% performance improvement** over traditional statistical baselines across multiple metrics
- Comprehensive evaluation using MSE, MAE, RMSE, R², and directional accuracy
- Results reflect realistic performance on efficient financial markets
- Baseline comparison methodology validates deep learning approach
- Framework enables fair, normalized comparison across all methods

### Technical Achievements

- Successfully trained 3-layer stacked LSTM on 25 years of financial data
- Implemented memory-efficient batch processing for large-scale time series
- Built comprehensive evaluation pipeline comparing 4 different forecasting approaches
- Demonstrated production-ready deployment with GPU acceleration (6-min training)
- Established proper ML evaluation principles through systematic baseline comparison

This project showcases rigorous ML methodology, honest evaluation, and understanding of when complex models provide measurable value over simpler alternatives.

## Key Components

### DataPreprocessor
Handles data loading, preprocessing, and sequence generation for time series forecasting.

### ForecastModel
Implements LSTM neural network with multiple prediction modes and training strategies.

### TraditionalForecasters
Provides baseline forecasting methods including naive, moving average, and ARIMA approaches.

### Performance Evaluation
Comprehensive metrics including directional accuracy for financial applications.

## API Reference

### DataPreprocessor Methods
- `prepare_train_sequences()`: Generate training sequences
- `prepare_test_sequences()`: Generate test sequences
- `create_train_batches()`: Memory-efficient batch generation
- `scale_data()`: Data normalization

### ForecastModel Methods
- `construct_network()`: Build LSTM architecture
- `train_model()`: Train with in-memory data
- `train_with_generator()`: Memory-efficient training
- `forecast_single_step()`: Point-by-point prediction
- `forecast_multi_sequence()`: Multiple sequence prediction

### TraditionalForecasters Methods
- `last_value_forecast()`: Naive forecasting
- `rolling_mean_forecast()`: Moving average forecasting
- `statistical_arima_forecast()`: ARIMA forecasting
- `run_all_methods()`: Execute all baseline methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - Feel free to use and modify for your projects.

## Contact

Created as part of machine learning portfolio development.

## Acknowledgments

- S&P 500 data from Yahoo Finance
- Built with TensorFlow/Keras
- Statistical methods via Statsmodels
- Optimized for Google Colab execution