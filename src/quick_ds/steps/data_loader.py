# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pathlib import Path
from typing import Annotated

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    random_state: int,
    is_inference: bool = False,
    targets: list[str] | None = None,
    path: str | None = None,
) -> Annotated[pd.DataFrame, "dataset"]:
    """Dataset reader step.

    This is an example of a dataset reader step that load Breast Cancer dataset.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured with number of rows and logic
    to drop target column or not. See the documentation for more information:

        https://docs.zenml.io/concepts/steps_and_pipelines#pipeline-parameterization

    Args:
        random_state: Random state for sampling
        is_inference: If `True` subset will be returned and target column
            will be removed from dataset.
        targets: Name of target columns in dataset.
        path: Path to dataset file (csv or parquet). If None, internal
            breast cancer dataset will be used.

    Returns:
        The dataset artifact as Pandas DataFrame and name of target column.
    """
    if path is None:
        path = Path(Path.getcwd(), "data", "01_raw")

    if not Path(path).exists():
        msg = f"Dataset not found at {path}"
        raise FileNotFoundError(msg)

    if Path(path).is_dir():
        files = Path(path).iterdir()
        csv_files = [f for f in files if f.endswith(".csv")]
        if len(csv_files) == 1:
            path = Path(path, csv_files[0])
        elif len(csv_files) > 1:
            msg = f"Multiple CSV files found in {path}. Please specify the file."
            raise ValueError(msg)

    if path.endswith(".csv"):
        dataset = pd.read_csv(path)
    elif path.endswith(".parquet") or Path(path).is_dir():
        dataset = pd.read_parquet(path)
    else:
        msg = f"Unsupported file format for path: {path}"
        raise ValueError(msg)

    inference_size = int(len(dataset) * 0.05)
    inference_subset = dataset.sample(inference_size, random_state=random_state)
    if is_inference:
        dataset = inference_subset
        for target in targets:
            if target in dataset.columns:
                dataset = dataset.drop(columns=target)
    else:
        dataset = dataset.drop(inference_subset.index)
    dataset = dataset.reset_index(drop=True)
    logger.info("Dataset with %d records loaded!", len(dataset))
    return dataset
