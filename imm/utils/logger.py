import sys
from loguru import logger
from pathlib import Path


def setup_logger(
    app_name,
    log_dir="logs",
    file_rotation="1 MB",
    file_level="DEBUG",
    console_level="INFO",
):
    """
    Set up the logger configuration.

    Parameters:
    - app_name: The name of the application to include in log messages.
    - log_dir: The directory where log files will be stored.
    - file_rotation: The rotation policy for the log files.
    - file_level: The logging level for the file logger.
    - console_level: The logging level for the console logger.
    """
    logger.remove()  # Remove the default logger

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Construct log file path based on app_name
    log_file = log_path / f"{app_name}_{{time}}.log"

    # Define log message formats
    console_format = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss}}</green> | <level>{{level: <8}}</level> | <cyan>{app_name}</cyan> | <level>{{message}}</level>"
    )
    file_format = f"{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {app_name} | {{message}}"

    # Configure loggers
    logger.add(log_file, rotation=file_rotation, level=file_level, format=file_format)  # Log to a file with rotation
    logger.add(sys.stdout, level=console_level, format=console_format)  # Log to stdout
