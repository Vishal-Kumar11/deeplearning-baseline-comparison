"""
TimeTracker: Performance timing utility for measuring execution duration

This class provides simple timing functionality to measure how long
operations take to execute. It's designed for performance monitoring
and benchmarking of machine learning operations.

Attributes:
    begin_datetime: The start time when timing begins
    begin_time: Alternative start time storage
    finish_time: The end time when timing stops

Example:
    timer = TimeTracker()
    timer.start_timing()
    # ... perform operation ...
    timer.stop_timing()
"""

import datetime as dt


class TimeTracker:
    """
    A utility class for measuring execution time of operations.
    
    This class provides methods to start and stop timing operations,
    automatically calculating and displaying the duration.
    """

    def __init__(self):
        """
        Initialize the TimeTracker instance.
        
        Sets up the timing state variables to track start and end times.
        """
        self.begin_datetime = None
        self.begin_time = None
        self.finish_time = None

    def start_timing(self):
        """
        Start timing an operation.
        
        Records the current datetime as the start time for timing calculations.
        This method should be called before the operation to be timed.
        """
        self.begin_datetime = dt.datetime.now()
        self.begin_time = self.begin_datetime

    def stop_timing(self):
        """
        Stop timing and display the duration.
        
        Calculates the elapsed time from when start_timing() was called
        and prints the duration in a human-readable format.
        
        Returns:
            datetime.timedelta: The duration of the timed operation
        """
        end_dt = dt.datetime.now()
        self.finish_time = end_dt
        duration = end_dt - self.begin_datetime
        print('Time taken: %s' % duration)
        return duration
