"""
Time-series split utilities.

We avoid random train/test splits to prevent lookahead bias in financial data.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd


@dataclass(frozen=True)
class RollingSplit:
    """
    Rolling / walk-forward split configuration.

    Example:
      - train_window_days = 365*4 (4 years)
      - test_window_days  = 365   (1 year)
      - step_days         = 90    (advance window quarterly)

    We split using dates (not row counts) to respect the time series nature.
    """
    train_window_days: int
    test_window_days: int
    step_days: int


def rolling_time_splits(
    df: pd.DataFrame,
    date_col: str,
    cfg: RollingSplit
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate rolling time window boundaries (train_start, train_end, test_start, test_end).

    Args:
        df: DataFrame containing a date column.
        date_col: Name of the date column (must be datetime-like).
        cfg: RollingSplit configuration.

    Yields:
        Tuple of (train_start, train_end, test_start, test_end) as Timestamps.
    """
    # Ensure datetime
    dates = pd.to_datetime(df[date_col])
    start = dates.min().normalize()
    end = dates.max().normalize()

    # We roll using actual calendar days.
    train_start = start
    train_end = train_start + pd.Timedelta(days=cfg.train_window_days)
    test_start = train_end
    test_end = test_start + pd.Timedelta(days=cfg.test_window_days)

    while test_end <= end:
        yield train_start, train_end, test_start, test_end

        # Advance windows
        train_start = train_start + pd.Timedelta(days=cfg.step_days)
        train_end = train_start + pd.Timedelta(days=cfg.train_window_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=cfg.test_window_days)
