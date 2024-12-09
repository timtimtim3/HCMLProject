import logging
from datetime import datetime
import os

LOG_DIR = "output/logs"


def setup_logger():
    """
    Set up a logger for the application with both file and console handlers.

    The logger writes messages to a log file and also outputs them to the console.
    Log files are named using a timestamp to ensure uniqueness.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{LOG_DIR}/training_{timestamp}.log'

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Remove the default handler to prevent duplicate logs
    logger.removeHandler(logger.handlers[0])

    return logger