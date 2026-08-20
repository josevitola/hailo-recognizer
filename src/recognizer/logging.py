import logging

LOGGER_NAME = "recognizer"


def setup_logging(debug: bool = False) -> None:
    """Configure project-wide logging for CLI entry points.

    With ``debug=False``, INFO messages are shown and DEBUG messages are
    suppressed. With ``debug=True``, DEBUG messages are also displayed.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove any pre-existing handlers so reconfiguration is idempotent
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the 'recognizer' package logger."""
    logging.basicConfig(filename="app.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)