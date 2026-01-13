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

from uuid import UUID

from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from quick_ds.pipelines import (
    feature_engineering,
)
from quick_ds.steps import (
    MODEL_OPTIONS,
    classifier_model_evaluator,
    model_promoter,
    model_trainer,
    regressor_model_evaluator,
)

logger = get_logger(__name__)

EVALUATOR = {
    "classifier": classifier_model_evaluator,
    "regressor": regressor_model_evaluator,
}


@pipeline
def training(
    train_dataset_id: UUID | None = None,
    test_dataset_id: UUID | None = None,
    targets: list[str] | None = None,
    drop_columns: list[str] | None = None,
    model_type: str | None = "sgd",
    dataset_path: str | None = None,
    score_thold: float = 0.5,
):
    """
    Model training pipeline.

    This is a pipeline that loads the data from a preprocessing pipeline,
    trains a model on it and evaluates the model. If it is the first model
    to be trained, it will be promoted to production. If not, it will be
    promoted only if it has a higher accuracy than the current production
    model version.

    Args:
        train_dataset_id: ID of the train dataset produced by feature engineering.
        test_dataset_id: ID of the test dataset produced by feature engineering.
        targets: Name of target columns in dataset.
        model_type: The type of model to train.
        dataset_path: Path to dataset file (csv or parquet).
        score_thold: Threshold for model promotion.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    # Execute Feature Engineering Pipeline
    if (
        train_dataset_id is None or test_dataset_id is None
    ) and dataset_path is not None:
        dataset_trn, dataset_tst = feature_engineering(
            drop_columns=drop_columns, targets=targets, dataset_path=dataset_path
        )
    else:
        client = Client()
        dataset_trn = client.get_artifact_version(name_id_or_prefix=train_dataset_id)
        dataset_tst = client.get_artifact_version(name_id_or_prefix=test_dataset_id)

    model = model_trainer(
        dataset_trn=dataset_trn, targets=targets, model_type=model_type
    )

    model_class = next(k for k, v in MODEL_OPTIONS.items() if model_type in v)
    model_evaluator = EVALUATOR[model_class]

    score = model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        targets=targets,
    )

    model_promoter(model_class=model_class, score=score, score_thold=score_thold)
