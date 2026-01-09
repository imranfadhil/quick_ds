from pathlib import Path

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_saver(
    dataset: pd.DataFrame,
    path: str,
    filename: str,
) -> None:
    """Saves a dataset to a specific path."""
    if not Path(path).exists():
        Path.mkdir(path, exist_ok=True, parents=True)

    full_path = Path(path, filename)
    if filename.endswith(".csv"):
        dataset.to_csv(full_path, index=False)
    elif filename.endswith(".parquet"):
        dataset.to_parquet(full_path, index=False)
    else:
        logger.warning("Unsupported file extension for %s. Saving as csv.", filename)
        dataset.to_csv(full_path + ".csv", index=False)

    logger.info("Saved dataset to %s", full_path)
