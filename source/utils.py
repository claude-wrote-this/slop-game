import threading
import logging
import datetime
import source.config as config
import os

class SingletonMeta(type):
    """A thread-safe metaclass for generating Singletons."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking pattern for thread safety
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    # super().__call__ handles both __new__ and __init__
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
        
class SingletonLoggingHandler(metaclass=SingletonMeta):
	_logger = None
	def __init__(self, level=logging.DEBUG, file=None):
		if file is None:
			file = os.path.join(config.PROJECT_ROOT, f"logs/{config.TITLE}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
		self._logger = logging.getLogger(f"{config.TITLE} Runtime Log")
		self._logger.setLevel(level)
		
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		console_formatter = logging.Formatter('%(levelname)s: %(message)s')
		console_handler.setFormatter(console_formatter)
		
		file_handler = logging.FileHandler(file)
		file_handler.setLevel(logging.DEBUG)
		file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
		file_handler.setFormatter(file_formatter)
		
		self._logger.addHandler(console_handler)
		self._logger.addHandler(file_handler)
		
	def get(self):
		return self._logger
		
log = SingletonLoggingHandler().get()