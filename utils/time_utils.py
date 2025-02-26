import time
from functools import wraps
import logging

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record start time
        result = func(*args, **kwargs)    # Call the actual function
        end_time = time.perf_counter()    # Record end time
        duration = end_time - start_time  # Calculate duration
        logging.info(f"Execution Time of '{func.__name__}': {duration:.4f} seconds")
        return result
    return wrapper

