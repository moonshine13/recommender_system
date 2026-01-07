"""
Logging setup for the application.

This module configures a logger with both console (StreamHandler) and rotating
file (RotatingFileHandler) handlers. Use this logger across the project for 
consistent logging format and levels.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Ensure the logs directory exists
os.makedirs("./logs", exist_ok=True)  # <-- this line creates the folder if it doesn't exist

# Create logger
logger = logging.getLogger("recommender_app")
logger.setLevel(logging.INFO)  # set minimum level to log

# Rotating File Handler
file_handler = RotatingFileHandler(
    "./logs/app.log",  # log file
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=3,  # keep last 3 logs
)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)  # only INFO+ to file


# Stream Handler (console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.DEBUG)  # DEBUG+ to console

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
