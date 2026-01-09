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

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import f1_score, mean_absolute_error
from zenml import log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def classifier_model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    targets: list[str] | None = None,
) -> float:
    """Evaluate a trained model.

    This is an example of a model evaluation step that takes in a model artifact
    previously trained by another step in your pipeline, and a training
    and validation data set pair which it uses to evaluate the model's
    performance. The model metrics are then returned as step output artifacts
    (in this case, the model accuracy on the train and test set).

    The suggested step implementation also outputs some warnings if the model
    performance does not meet some minimum criteria. This is just an example of
    how you can use steps to monitor your model performance and alert you if
    something goes wrong. As an alternative, you can raise an exception in the
    step to force the pipeline run to fail early and all subsequent steps to
    be skipped.

    This step is parameterized to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use different values for the acceptable model performance thresholds and
    to control whether the pipeline run should fail if the model performance
    does not meet the minimum criteria. See the documentation for more
    information:

        https://docs.zenml.io/concepts/steps_and_pipelines#pipeline-parameterization

    Args:
        model: The pre-trained model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        targets: Name of target columns in dataset.

    Returns:
        The model F1 score on the test set.
    """
    # Calculate the model F1 score on the train and test set
    trn_pred = model.predict(dataset_trn.drop(columns=targets))
    trn_f1 = f1_score(dataset_trn[targets], trn_pred, average="weighted")
    tst_pred = model.predict(dataset_tst.drop(columns=targets))
    tst_f1 = f1_score(dataset_tst[targets], tst_pred, average="weighted")
    logger.info("Train F1=%.2f%%", trn_f1 * 100)
    logger.info("Test F1=%.2f%%", tst_f1 * 100)

    client = Client()
    latest_classifier = client.get_artifact_version("sklearn_classifier")

    log_metadata(
        metadata={
            "train_f1": float(trn_f1),
            "test_f1": float(tst_f1),
        },
        artifact_version_id=latest_classifier.id,
    )

    return float(tst_f1)


@step
def regressor_model_evaluator(
    model: RegressorMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    targets: list[str] | None = None,
) -> float:
    """Evaluate a trained regression model.

    Args:
        model: The pre-trained regression model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        targets: Name of target column in dataset.

    Returns:
        The model MAE on the test set.
    """
    # Calculate the model MAE on the train and test set
    trn_pred = model.predict(dataset_trn.drop(columns=targets))
    trn_mae = mean_absolute_error(dataset_trn[targets], trn_pred)
    tst_pred = model.predict(dataset_tst.drop(columns=targets))
    tst_mae = mean_absolute_error(dataset_tst[targets], tst_pred)
    logger.info("Train MAE=%.2f", trn_mae)
    logger.info("Test MAE=%.2f", tst_mae)

    client = Client()
    latest_model = client.get_artifact_version("sklearn_classifier")

    log_metadata(
        metadata={
            "train_mae": float(trn_mae),
            "test_mae": float(tst_mae),
        },
        artifact_version_id=latest_model.id,
    )

    return float(tst_mae)
