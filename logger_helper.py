import sys
import logging
import time
import os


def logger_setup(
    logfile: str = "./logs/training.log",
    experiment_logfile: str | None = None,
) -> None:
    """
    Set up root logger with up to three handlers:
      1. Global log file  — always ``logs/training.log``
      2. Stdout stream
      3. Experiment-specific log file (optional)

    Parameters
    ----------
    logfile : str
        Path to the global log file (default: ``./logs/training.log``).
    experiment_logfile : str | None
        If provided, an additional file handler is created for an
        experiment-specific log (e.g. ``logs/logs_rubsheet_full.log``).
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)15s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"  # ISO 8601; UTC

    formatter = logging.Formatter(fmt, datefmt)
    formatter.converter = time.gmtime  # force UTC

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # avoid re-adding handlers on hot-reload
    if root.handlers:
        return

    # --- Global log file (logs/training.log) ---
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    file_h = logging.FileHandler(logfile, encoding="utf-8")
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    # --- Stream handler (stdout) ---
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)
    root.addHandler(stream_h)

    # --- Experiment-specific log file ---
    if experiment_logfile:
        os.makedirs(os.path.dirname(experiment_logfile), exist_ok=True)
        exp_h = logging.FileHandler(experiment_logfile, encoding="utf-8")
        exp_h.setFormatter(formatter)
        root.addHandler(exp_h)
