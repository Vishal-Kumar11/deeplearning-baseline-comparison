"""
DataPreprocessor: Comprehensive data handling and preprocessing for time series forecasting

This class handles loading, preprocessing, and batch generation for time series data.
It provides methods for creating training and testing sequences, normalizing data,
and generating batches for efficient memory usage during training.

Attributes:
    csv_path: Path to the CSV data file
    split_ratio: Ratio for train/test split
    column_names: List of column names to use from the dataset
    train_set: Training data array
    test_set: Testing data array
    train_size: Number of training samples
    test_size: Number of testing samples
    batch_count: Number of training batches available

Example:
    processor = DataPreprocessor('data/sp500.csv', 0.85, ['Close', 'Volume'])
    train_x, train_y = processor.prepare_train_sequences(50, True)
    test_x, test_y = processor.prepare_test_sequences(50, True)
"""

import math
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    A comprehensive class for loading and preprocessing time series data for LSTM models.
    
    This class handles data loading, train/test splitting, sequence generation,
    normalization, and batch creation for efficient training of neural networks.
    """

    def __init__(self, csv_path, split_ratio, column_names):
        """
        Initialize the DataPreprocessor with data file and configuration.
        
        Loads the CSV file, performs train/test split, and prepares data arrays
        for sequence generation and batch processing.
        
        Args:
            csv_path (str): Path to the CSV data file
            split_ratio (float): Ratio for train/test split (0.0 to 1.0)
            column_names (list): List of column names to extract from the dataset
            
        Raises:
            FileNotFoundError: If the CSV file cannot be found
            ValueError: If split_ratio is not between 0 and 1
        """
        # Load and validate the dataset
        dataframe = pd.read_csv(csv_path)
        
        # Calculate split point for train/test division
        i_split = int(len(dataframe) * split_ratio)
        
        # Extract specified columns and split into train/test sets
        self.train_set = dataframe.get(column_names).values[:i_split]
        self.test_set = dataframe.get(column_names).values[i_split:]
        
        # Store configuration parameters
        self.csv_path = csv_path
        self.split_ratio = split_ratio
        self.column_names = column_names
        
        # Calculate dataset sizes for efficient batch processing
        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)
        self.batch_count = None

    def prepare_test_sequences(self, seq_len, should_normalize):
        """
        Create input/output sequences for testing data.
        
        Generates overlapping windows of test data for model evaluation.
        Uses batch processing for memory efficiency with large datasets.
        
        Args:
            seq_len (int): Length of input sequences
            should_normalize (bool): Whether to normalize the data windows
            
        Returns:
            tuple: (input_sequences, output_values) arrays for testing
            
        Warning:
            This method loads all test data into memory. For very large datasets,
            consider using a generator-based approach to avoid memory issues.
        """
        test_sequences = []
        
        # Generate overlapping windows from test data
        for i in range(self.test_size - seq_len):
            test_sequences.append(self.test_set[i:i+seq_len])

        # Convert to numpy array for efficient processing
        test_sequences = np.array(test_sequences).astype(float)
        
        # Apply normalization if requested
        if should_normalize:
            test_sequences = self.scale_data(test_sequences, single_window=False)
        
        # Split into input sequences and target values
        input_sequences = test_sequences[:, :-1]
        output_values = test_sequences[:, -1, [0]]
        
        return input_sequences, output_values

    def prepare_train_sequences(self, seq_len, should_normalize):
        """
        Create input/output sequences for training data.
        
        Generates overlapping windows of training data for model training.
        Uses batch processing for memory efficiency with large datasets.
        
        Args:
            seq_len (int): Length of input sequences
            should_normalize (bool): Whether to normalize the data windows
            
        Returns:
            tuple: (input_sequences, output_values) arrays for training
            
        Warning:
            This method loads all training data into memory. For very large datasets,
            consider using create_train_batches() method for generator-based processing.
        """
        input_batches = []
        output_batches = []
        
        # Generate overlapping windows from training data
        for i in range(self.train_size - seq_len):
            input_seq, output_val = self._extract_window(i, seq_len, should_normalize)
            input_batches.append(input_seq)
            output_batches.append(output_val)
            
        return np.array(input_batches), np.array(output_batches)

    def create_train_batches(self, seq_len, batch_count, should_normalize):
        """
        Generate training data batches using a memory-efficient generator.
        
        Creates batches of training sequences on-demand to minimize memory usage.
        This is particularly useful for large datasets that don't fit in memory.
        
        Args:
            seq_len (int): Length of input sequences
            batch_count (int): Number of samples per batch
            should_normalize (bool): Whether to normalize the data windows
            
        Yields:
            tuple: (input_batches, output_batches) arrays for training
            
        Note:
            This generator will cycle through the training data indefinitely,
            making it suitable for training loops that require multiple epochs.
        """
        batch_index = 0
        
        while batch_index < (self.train_size - seq_len):
            input_batches = []
            output_batches = []
            
            # Create a batch of specified size
            for b in range(batch_count):
                if batch_index >= (self.train_size - seq_len):
                    # Handle smaller final batch if data doesn't divide evenly
                    yield np.array(input_batches), np.array(output_batches)
                    batch_index = 0
                    
                input_seq, output_val = self._extract_window(batch_index, seq_len, should_normalize)
                input_batches.append(input_seq)
                output_batches.append(output_val)
                batch_index += 1
                
            yield np.array(input_batches), np.array(output_batches)

    def _extract_window(self, batch_index, seq_len, should_normalize):
        """
        Extract a single data window from the training set.
        
        Creates a sequence window from the training data starting at the given index.
        This is a helper method used by both batch and non-batch data generation.
        
        Args:
            batch_index (int): Starting index for the window
            seq_len (int): Length of the sequence window
            should_normalize (bool): Whether to normalize the window
            
        Returns:
            tuple: (input_sequence, output_value) for the window
        """
        sequence_window = self.train_set[batch_index:batch_index+seq_len]
        
        # Apply normalization if requested
        if should_normalize:
            sequence_window = self.scale_data(sequence_window, single_window=True)[0]
            
        # Split into input sequence and target value
        input_seq = sequence_window[:-1]
        output_val = sequence_window[-1, [0]]
        
        return input_seq, output_val

    def scale_data(self, sequence_data, single_window=False):
        """
        Normalize data windows using a base value of zero.
        
        Applies normalization to data windows by dividing each value by the first
        value in the window and subtracting 1. This creates a percentage change
        representation that's often more suitable for neural network training.
        
        Args:
            sequence_data (numpy.ndarray): Data windows to normalize
            single_window (bool): Whether the input is a single window or multiple
            
        Returns:
            numpy.ndarray: Normalized data windows
            
        Note:
            This normalization method assumes the first value in each window
            represents a meaningful baseline for percentage calculations.
        """
        scaled_sequences = []
        
        # Handle single window case by wrapping in list
        if single_window:
            sequence_data = [sequence_data]
            
        # Process each window individually
        for window in sequence_data:
            scaled_seq = []
            
            # Normalize each column separately
            for column_idx in range(window.shape[1]):
                # Calculate percentage change from first value (with epsilon to prevent division by zero)
                scaled_column = [((float(p) / (float(window[0, column_idx]) + 1e-8)) - 1) 
                               for p in window[:, column_idx]]
                scaled_seq.append(scaled_column)
                
            # Reshape and transpose back to original format
            scaled_seq = np.array(scaled_seq).T
            scaled_sequences.append(scaled_seq)
            
        return np.array(scaled_sequences)
