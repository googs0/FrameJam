import logging
import logging.config
from colorama import init, Fore, Style
import inspect
import cv2

init()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.MAGENTA,
        'INFO': Fore.LIGHTGREEN_EX,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Style.BRIGHT + Fore.RED,
    }

    RESET = Style.RESET_ALL

    def format(self, record):
        log_message = super().format(record)
        log_level = record.levelname

        # Add color to log messages based on log level
        if log_level in self.COLORS:
            log_message = f"{self.COLORS[log_level]}{log_message}{self.RESET}"

        return log_message


def setup_logging():
    # _main.py logger
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.DEBUG)

    # unit_test.py logger
    opencv_logger = logging.getLogger('cv2')

    # Check if the setup_logging function is called from unit_test.py
    calling_module = inspect.getmodule(inspect.currentframe().f_back)
    if calling_module and calling_module.__name__ == 'unit_test':
        cv2.setLogLevel(0)
    else:
        cv2.setLogLevel(1)

    # If you want to redirect OpenCV logs to a separate file, you can add a FileHandler for the OpenCV logger
    opencv_file_handler = logging.FileHandler('opencv_log_file.log')
    opencv_file_handler.setLevel(logging.INFO)  # Set the desired logging level for the file handler
    opencv_file_handler.setFormatter(formatter)
    opencv_logger.addHandler(opencv_file_handler)
