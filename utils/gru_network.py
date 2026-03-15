"""
GRUForecastModel: GRU neural network for time series forecasting

This class mirrors ForecastModel but uses GRU (Gated Recurrent Unit) layers
instead of LSTM. GRU is a simpler recurrent architecture that often matches
LSTM performance with fewer parameters and faster training.

Attributes:
    network: The compiled Keras Sequential model

Example:
    model = GRUForecastModel()
    model.construct_network(config_dict)
    model.train_with_generator(generator, epochs, batch_count, batches_per_epoch, checkpoint_dir)
    predictions = model.forecast_multi_sequence(test_inputs, seq_len, forecast_horizon)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import datetime as dt
import numpy as np
from numpy import newaxis
from utils.timer import TimeTracker
from keras.layers import Dense, Dropout, GRU
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


class GRUForecastModel:
    """
    A GRU-based forecasting model for time series prediction.

    Provides the same interface as ForecastModel but uses GRU layers,
    enabling a direct apples-to-apples comparison between LSTM and GRU
    architectures on the same dataset and training regime.
    """

    def __init__(self):
        """Initialize the GRUForecastModel with an empty Sequential network."""
        self.network = Sequential()

    def construct_network(self, config_dict):
        """
        Build the GRU network architecture from configuration.

        Reads the 'gru_architecture' key from config_dict, mirroring the
        same layer schema used by the LSTM config but substituting GRU layers.

        Args:
            config_dict (dict): Configuration dictionary with 'gru_architecture' key
        """
        timer = TimeTracker()
        timer.start_timing()

        for layer in config_dict['gru_architecture']['layers']:
            num_neurons  = layer.get('neurons')
            dropout_prob = layer.get('rate')
            activation   = layer.get('activation')
            return_seq   = layer.get('return_seq')
            time_steps   = layer.get('input_timesteps')
            num_features = layer.get('input_dim')

            if layer['type'] == 'gru':
                self.network.add(GRU(
                    num_neurons,
                    input_shape=(time_steps, num_features),
                    return_sequences=return_seq
                ))
            elif layer['type'] == 'dropout':
                self.network.add(Dropout(dropout_prob))
            elif layer['type'] == 'dense':
                self.network.add(Dense(num_neurons, activation=activation))

        self.network.compile(
            loss=config_dict['gru_architecture']['loss_function'],
            optimizer=config_dict['gru_architecture']['optimization_method']
        )

        print('[GRUForecastModel] Network Compiled')
        timer.stop_timing()

    def train_with_generator(self, batch_generator, num_epochs, batch_count,
                             batches_per_epoch, checkpoint_dir):
        """
        Train the GRU model using a memory-efficient data generator.

        Args:
            batch_generator (generator): Yields (inputs, targets) batches
            num_epochs (int): Number of training epochs
            batch_count (int): Batch size
            batches_per_epoch (int): Steps per epoch
            checkpoint_dir (str): Directory to save model checkpoints

        Returns:
            str: Path to the saved model checkpoint
        """
        timer = TimeTracker()
        timer.start_timing()
        print('[GRUForecastModel] Training Started')
        print('[GRUForecastModel] %s epochs, %s batch size, %s batches per epoch' %
              (num_epochs, batch_count, batches_per_epoch))

        checkpoint_path = os.path.join(
            checkpoint_dir,
            'gru-%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), num_epochs)
        )

        callbacks = [ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True)]

        self.network.fit(
            batch_generator,
            steps_per_epoch=batches_per_epoch,
            epochs=num_epochs,
            callbacks=callbacks
        )

        print('[GRUForecastModel] Training Completed. Model saved as %s' % checkpoint_path)
        timer.stop_timing()
        return checkpoint_path

    def forecast_multi_sequence(self, input_data, sequence_len, forecast_horizon):
        """
        Predict multiple sequences using batched inference.

        Uses the same batched approach as ForecastModel.forecast_multi_sequence
        to keep predict() calls equal to forecast_horizon rather than
        n_sequences * forecast_horizon.

        Args:
            input_data (numpy.ndarray): Test input sequences
            sequence_len (int): Length of input sequences
            forecast_horizon (int): Number of steps to predict ahead

        Returns:
            list: List of prediction sequences
        """
        print('[GRUForecastModel] Predicting Sequences Multiple...')
        n_sequences = int(len(input_data) / forecast_horizon)
        n_features = input_data.shape[2]

        windows = np.array([input_data[i * forecast_horizon] for i in range(n_sequences)])
        forecast_array = np.zeros((n_sequences, forecast_horizon))

        for j in range(forecast_horizon):
            preds = self.network.predict(windows, verbose=0)[:, 0]
            forecast_array[:, j] = preds
            new_rows = np.broadcast_to(
                preds[:, newaxis, newaxis], (n_sequences, 1, n_features)
            ).copy()
            windows = np.concatenate([windows[:, 1:, :], new_rows], axis=1)

        return forecast_array.tolist()
