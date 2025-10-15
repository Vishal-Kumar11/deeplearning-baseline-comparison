"""
ForecastModel: Advanced LSTM neural network for time series forecasting

This class provides a comprehensive LSTM-based forecasting model with support for
various prediction modes, training strategies, and model persistence. It handles
network construction, training, and inference for time series data.

Attributes:
    network: The compiled Keras Sequential model
    config_dict: Configuration dictionary containing model parameters
    layer_params: Parameters for individual network layers
    num_neurons: Number of neurons in LSTM layers
    dropout_prob: Dropout probability for regularization
    activation_fn: Activation function for dense layers
    return_seq: Whether LSTM layers return sequences
    time_steps: Number of input timesteps
    num_features: Number of input features
    train_inputs: Training input data
    train_targets: Training target data
    num_epochs: Number of training epochs
    batch_count: Batch size for training
    checkpoint_dir: Directory for saving model checkpoints
    checkpoint_path: Full path for model checkpoint file
    model_callbacks: Training callbacks for model optimization
    batch_generator: Data generator for batch training
    batches_per_epoch: Number of batches per training epoch
    forecast_values: Model predictions
    input_window: Current input window for prediction
    forecast_list: List of forecasted values
    sequence_len: Length of input sequences
    forecast_horizon: Number of steps to forecast ahead
    forecast_sequence: Complete forecast sequence

Example:
    model = ForecastModel()
    model.construct_network(config_dict)
    model.train_model(train_x, train_y, epochs=50, batch_size=32, save_dir='models')
    predictions = model.forecast_single_step(test_data)
"""

# Fix TensorFlow threading issues on macOS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import math
import numpy as np
import datetime as dt
from numpy import newaxis
from utils.timer import TimeTracker
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class ForecastModel:
    """
    A comprehensive LSTM-based forecasting model for time series prediction.
    
    This class provides methods for building, training, and using LSTM neural networks
    for time series forecasting. It supports various prediction modes and training
    strategies optimized for time series data.
    """

    def __init__(self):
        """
        Initialize the ForecastModel instance.
        
        Creates a new Sequential model instance for building the LSTM network.
        """
        self.network = Sequential()

    def load_saved_model(self, filepath):
        """
        Load a previously saved model from disk.
        
        Loads a trained model from the specified file path for inference or
        continued training. This is useful for model persistence and deployment.
        
        Args:
            filepath (str): Path to the saved model file (.h5 format)
            
        Raises:
            FileNotFoundError: If the model file cannot be found
            ValueError: If the model file is corrupted or incompatible
        """
        print('[ForecastModel] Loading model from file %s' % filepath)
        self.network = load_model(filepath)

    def construct_network(self, config_dict):
        """
        Build the LSTM network architecture based on configuration.
        
        Constructs a sequential neural network with LSTM layers, dropout regularization,
        and dense output layers according to the provided configuration. The network
        is compiled with the specified loss function and optimizer.
        
        Args:
            config_dict (dict): Configuration dictionary containing model parameters
                Expected keys: 'architecture' with 'layers' and training parameters
                
        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If layer configuration is invalid
        """
        timer = TimeTracker()
        timer.start_timing()

        # Process each layer configuration
        for layer in config_dict['architecture']['layers']:
            # Extract layer parameters with defaults
            num_neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_prob = layer['rate'] if 'rate' in layer else None
            activation_fn = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            time_steps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            num_features = layer['input_dim'] if 'input_dim' in layer else None

            # Add layers based on type
            if layer['type'] == 'dense':
                self.network.add(Dense(num_neurons, activation=activation_fn))
            elif layer['type'] == 'lstm':
                self.network.add(LSTM(num_neurons, 
                                    input_shape=(time_steps, num_features), 
                                    return_sequences=return_seq))
            elif layer['type'] == 'dropout':
                self.network.add(Dropout(dropout_prob))

        # Compile the network with specified loss and optimizer
        self.network.compile(loss=config_dict['architecture']['loss_function'], 
                           optimizer=config_dict['architecture']['optimization_method'])

        print('[ForecastModel] Network Compiled')
        timer.stop_timing()

    def train_model(self, train_inputs, train_targets, num_epochs, batch_count, checkpoint_dir):
        """
        Train the LSTM model using in-memory data.
        
        Trains the model using the provided training data with early stopping
        and model checkpointing for optimal performance and model persistence.
        
        Args:
            train_inputs (numpy.ndarray): Training input sequences
            train_targets (numpy.ndarray): Training target values
            num_epochs (int): Number of training epochs
            batch_count (int): Batch size for training
            checkpoint_dir (str): Directory to save model checkpoints
            
        Returns:
            str: Path to the saved model file
            
        Note:
            This method loads all training data into memory. For large datasets,
            consider using train_with_generator() for memory-efficient training.
        """
        timer = TimeTracker()
        timer.start_timing()
        print('[ForecastModel] Training Started')
        print('[ForecastModel] %s epochs, %s batch size' % (num_epochs, batch_count))
        
        # Create checkpoint filename with timestamp
        checkpoint_path = os.path.join(checkpoint_dir, '%s-e%s.h5' % 
                                     (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(num_epochs)))
        
        # Set up training callbacks
        model_callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        self.network.fit(
            train_inputs,
            train_targets,
            epochs=num_epochs,
            batch_size=batch_count,
            callbacks=model_callbacks
        )
        
        # Save the final model
        self.network.save(checkpoint_path)

        print('[ForecastModel] Training Completed. Model saved as %s' % checkpoint_path)
        timer.stop_timing()
        
        return checkpoint_path

    def train_with_generator(self, batch_generator, num_epochs, batch_count, batches_per_epoch, checkpoint_dir):
        """
        Train the LSTM model using a data generator for memory efficiency.
        
        Trains the model using a generator that yields batches of data, allowing
        for training on datasets that are too large to fit in memory.
        
        Args:
            batch_generator (generator): Data generator yielding (inputs, targets) batches
            num_epochs (int): Number of training epochs
            batch_count (int): Batch size for training
            batches_per_epoch (int): Number of batches per epoch
            checkpoint_dir (str): Directory to save model checkpoints
            
        Returns:
            str: Path to the saved model file
            
        Note:
            This method is memory-efficient and suitable for large datasets
            that cannot be loaded entirely into memory.
        """
        timer = TimeTracker()
        timer.start_timing()
        print('[ForecastModel] Training Started')
        print('[ForecastModel] %s epochs, %s batch size, %s batches per epoch' % 
              (num_epochs, batch_count, batches_per_epoch))
        
        # Create checkpoint filename with timestamp
        checkpoint_path = os.path.join(checkpoint_dir, '%s-e%s.h5' % 
                                     (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(num_epochs)))
        
        # Set up training callbacks
        model_callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True)
        ]
        
        # Train the model using generator
        self.network.fit(
            batch_generator,
            steps_per_epoch=batches_per_epoch,
            epochs=num_epochs,
            callbacks=model_callbacks
        )
        
        print('[ForecastModel] Training Completed. Model saved as %s' % checkpoint_path)
        timer.stop_timing()
        
        return checkpoint_path

    def forecast_single_step(self, input_data):
        """
        Predict each timestep given the last sequence of true data.
        
        Performs point-by-point prediction where each prediction is based on
        the previous sequence of true data. This is useful for one-step-ahead
        forecasting with high accuracy.
        
        Args:
            input_data (numpy.ndarray): Input sequences for prediction
            
        Returns:
            numpy.ndarray: Flattened array of predictions
            
        Note:
            This method predicts one step ahead at a time, making it suitable
            for short-term forecasting with high accuracy requirements.
        """
        print('[ForecastModel] Predicting Point-by-Point...')
        forecast_values = self.network.predict(input_data)
        forecast_values = np.reshape(forecast_values, (forecast_values.size,))
        return forecast_values

    def forecast_multi_sequence(self, input_data, sequence_len, forecast_horizon):
        """
        Predict multiple sequences by shifting the prediction window.
        
        Generates multiple prediction sequences by shifting the input window
        forward by the forecast horizon after each prediction. This is useful
        for medium-term forecasting with multiple prediction points.
        
        Args:
            input_data (numpy.ndarray): Input sequences for prediction
            sequence_len (int): Length of input sequences
            forecast_horizon (int): Number of steps to predict ahead
            
        Returns:
            list: List of prediction sequences
            
        Note:
            This method creates multiple prediction sequences by shifting
            the input window, making it suitable for medium-term forecasting.
        """
        print('[ForecastModel] Predicting Sequences Multiple...')
        forecast_list = []
        
        # Generate multiple prediction sequences
        for i in range(int(len(input_data)/forecast_horizon)):
            input_window = input_data[i*forecast_horizon]
            forecast_sequence = []
            
            # Predict multiple steps ahead
            for j in range(forecast_horizon):
                forecast_sequence.append(self.network.predict(input_window[newaxis,:,:])[0,0])
                input_window = input_window[1:]
                input_window = np.insert(input_window, [sequence_len-2], forecast_sequence[-1], axis=0)
                
            forecast_list.append(forecast_sequence)
            
        return forecast_list

    def forecast_full_sequence(self, input_data, sequence_len):
        """
        Predict the full sequence by shifting the window continuously.
        
        Generates predictions for the entire input sequence by continuously
        shifting the input window and making predictions. This is useful
        for long-term forecasting and trend analysis.
        
        Args:
            input_data (numpy.ndarray): Input sequences for prediction
            sequence_len (int): Length of input sequences
            
        Returns:
            list: Complete sequence of predictions
            
        Note:
            This method provides continuous predictions by shifting the window
            one step at a time, making it suitable for long-term forecasting.
        """
        print('[ForecastModel] Predicting Sequences Full...')
        input_window = input_data[0]
        forecast_sequence = []
        
        # Predict for the entire sequence
        for i in range(len(input_data)):
            forecast_sequence.append(self.network.predict(input_window[newaxis,:,:])[0,0])
            input_window = input_window[1:]
            input_window = np.insert(input_window, [sequence_len-2], forecast_sequence[-1], axis=0)
            
        return forecast_sequence
