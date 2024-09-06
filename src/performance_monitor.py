import time
import psutil
import logging
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()

    def log_performance(self):
        current_time = time.time()
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()

        logging.info(f"Performance: Time: {current_time - self.start_time:.2f}s, "
                     f"CPU: {cpu_percent}%, "
                     f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB")

def performance_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        result = func(*args, **kwargs)
        monitor.log_performance()
        return result
    return wrapper