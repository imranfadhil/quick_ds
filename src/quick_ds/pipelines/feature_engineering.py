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

from zenml import pipeline
from zenml.logger import get_logger

from quick_ds.steps import (
    data_loader,
    data_preprocessor,
    data_saver,
    data_splitter,
)

logger = get_logger(__name__)


@pipeline
def feature_engineering(
    test_size: float = 0.2,
    drop_na: bool | None = None,
    normalize: bool | None = None,
    drop_columns: list[str] | None = None,
    targets: list[str] | None = None,
    random_state: int = 17,
    dataset_path: str | None = None,
    output_path: str | None = None,
):
    """
    Feature engineering pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets.

    Args:
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        targets: Name of target columns in dataset
        random_state: Random state to configure the data loader
        dataset_path: Path to dataset file (csv or parquet).
        output_path: Path to save the processed datasets.

    Returns:
        The processed datasets (dataset_trn, dataset_tst).
    """
    if dataset_path:
        dataset_path = (
            dataset_path
            if Path(dataset_path).is_absolute()
            else str(Path.cwd() / dataset_path)
        )
    if output_path:
        output_path = (
            output_path
            if Path(output_path).is_absolute()
            else str(Path.cwd() / output_path)
        )
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    raw_data = data_loader(
        random_state=random_state, targets=targets, path=dataset_path
    )
    dataset_trn, dataset_tst = data_splitter(
        dataset=raw_data,
        test_size=test_size,
    )
    dataset_trn, dataset_tst, _ = data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        drop_na=drop_na,
        normalize=normalize,
        drop_columns=drop_columns,
        targets=targets,
        random_state=random_state,
    )

    if output_path:
        data_saver(
            dataset=dataset_trn,
            path=output_path,
            filename="dataset_trn.csv",
        )
        data_saver(
            dataset=dataset_tst,
            path=output_path,
            filename="dataset_tst.csv",
        )
    return dataset_trn, dataset_tst
