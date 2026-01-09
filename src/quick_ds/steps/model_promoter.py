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

from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(
    model_class: str, score: float, score_thold: float, stage: str = "production"
) -> bool:
    """Model promoter step.

    This is an example of a step that conditionally promotes a model. It takes
    in the score of the model and the stage to promote the model to. If the
    score is below threshold, the model is not promoted. If it is above, the
    model is promoted to the stage indicated in the parameters. If there is
    already a model in the indicated stage, the model with the better score
    is promoted.

    Args:
        model_class: classifier or regressor.
        score: Score of the model (F1 for classifier, MAE for regressor).
        score_thold: Threshold for model promotion.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    is_promoted = False
    is_better = (
        score > score_thold if model_class == "classifier" else score < score_thold
    )

    if not is_better:
        msg = (
            (
                "Model F1 score %.2f%% is below %s ! Not promoting model.",
                score * 100,
                score_thold,
            )
            if model_class == "classifier"
            else (
                "Model MAE score %.2f%% is more than %s ! Not promoting model.",
                score * 100,
                score_thold,
            )
        )
        logger.info(msg)
    else:
        logger.info("Model promoted to %s!", stage)
        is_promoted = True

        # Get the model in the current context
        current_model = get_step_context().model

        # Get the model that is in the production stage
        client = Client()
        try:
            stage_model = client.get_model_version(current_model.name, stage)
            # We compare their metrics
            prod_score = (
                stage_model.get_artifact("sklearn_classifier").run_metadata["test_f1"]
                if model_class == "classifier"
                else stage_model.get_artifact("sklearn_regressor").run_metadata[
                    "test_mae"
                ]
            )
            if float(score) > float(prod_score):
                # If current model has better metrics, we promote it
                is_promoted = True
                current_model.set_stage(stage, force=True)
        except KeyError:
            # If no such model exists, current one is promoted
            is_promoted = True
            current_model.set_stage(stage, force=True)
    return is_promoted
