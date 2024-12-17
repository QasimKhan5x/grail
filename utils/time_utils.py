import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record start time
        result = func(*args, **kwargs)    # Call the actual function
        end_time = time.perf_counter()    # Record end time
        duration = end_time - start_time  # Calculate duration
        print(f"Execution Time of '{func.__name__}': {duration:.4f} seconds")
        return result
    return wrapper

