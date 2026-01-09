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
import shutil
import sys
from importlib import metadata
from pathlib import Path
from subprocess import Popen

import click
import yaml
from dotenv import load_dotenv
from zenml.client import Client
from zenml.logger import get_logger

from quick_ds.pipelines import (
    feature_engineering,
    inference,
    training,
)

logger = get_logger(__name__)
load_dotenv()

try:
    quick_dsVersion = metadata.version("quick_ds")
except metadata.PackageNotFoundError:
    quick_dsVersion = "0.0.0"


FRONTEND_DIR = Path(__file__).parent / "apps" / "ui"


@click.group()
@click.version_option(
    prog_name="quick_ds",
    version=quick_dsVersion,
    message="%(prog)s version: %(version)s",
)
def cli():
    """
    Command-line interface for the quick_ds library.

    Provides commands to run the web application, manage MLflow servers,
    and execute machine learning pipelines for training and prediction.
    """
    pass  # noqa: PIE790


@click.command(
    help="""
ZenML Starter project.

Run the ZenML starter project with basic options.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the training pipeline
    python run.py --training-pipeline

  \b 
  # Run the training pipeline with versioned artifacts
    python run.py --training-pipeline --train-dataset-version-name=1 --test-dataset-version-name=1

  \b
  # Run the inference pipeline
    python run.py --inference-pipeline

  \b
  # Deploy a model locally with FastAPI
    python run.py --deploy-locally --deployment-model-name=my_model

"""
)
@click.option(
    "--train-dataset-name",
    default="dataset_trn",
    type=click.STRING,
    help="The name of the train dataset produced by feature engineering.",
)
@click.option(
    "--train-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the train dataset produced by feature engineering. "
    "If not specified, a new version will be created.",
)
@click.option(
    "--test-dataset-name",
    default="dataset_tst",
    type=click.STRING,
    help="The name of the test dataset produced by feature engineering.",
)
@click.option(
    "--test-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the test dataset produced by feature engineering. "
    "If not specified, a new version will be created.",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--train-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)
@click.option(
    "--use-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--raw-data-path",
    default="data/01_raw",
    help="Path to raw data for feature engineering.",
)
@click.option(
    "--intermediate-data-path",
    default="data/02_intermediate",
    help="Path to save intermediate data from feature engineering.",
)
@click.option(
    "--primary-data-path",
    default="data/03_primary",
    help="Path to primary data for training.",
)
def zenml(
    train_dataset_name: str = "dataset_trn",
    train_dataset_version_name: str | None = None,
    test_dataset_name: str = "dataset_tst",
    test_dataset_version_name: str | None = None,
    feature_pipeline: bool = False,
    train_pipeline: bool = False,
    inference_pipeline: bool = False,
    use_cache: bool = False,
    raw_data_path: str = "data/01_raw",
    intermediate_data_path: str = "data/02_intermediate",
    primary_data_path: str = "data/03_primary",
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        train_dataset_name: The name of the train dataset produced by feature engineering.
        train_dataset_version_name: Version of the train dataset produced by feature engineering.
            If not specified, a new version will be created.
        test_dataset_name: The name of the test dataset produced by feature engineering.
        test_dataset_version_name: Version of the test dataset produced by feature engineering.
            If not specified, a new version will be created.
        feature_pipeline: Whether to run the pipeline that creates the dataset.
        train_pipeline: Whether to run the pipeline that trains the model.
        inference_pipeline: Whether to run the pipeline that performs inference.
        use_cache: If `True` cache will be used.
        raw_data_path: Path to raw data.
        intermediate_data_path: Path to save intermediate data.
        primary_data_path: Path to primary data.
        deploy_locally: Whether to run the pipeline that deploys a model locally with FastAPI.
        deployment_model_name: Name of the model to deploy locally.
        deployment_model_stage: Stage of the model to deploy.
        deployment_model_artifact_name: Name of the model artifact to load.
        deployment_preprocess_pipeline_name: Name of the preprocessing pipeline artifact to load.
        deployment_port: Port to expose the deployment server on.
        deployment_zenml_server: URL of the ZenML server for deployment.
        deployment_zenml_api_key: API key for the ZenML server.
    """
    client = Client()

    source_config_folder = Path(__file__).resolve().parent / "configs"
    config_folder = Path.cwd() / "configs"

    if not config_folder.exists():
        logger.info("Configs folder not found in CWD. Copying default configs...")
        shutil.copytree(source_config_folder, config_folder)
        logger.info("Configs copied to %s", config_folder)

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        pipeline_args = {}
        if not use_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = Path(config_folder, "feature_engineering.yaml")
        run_args_feature = {
            "dataset_path": raw_data_path,
            "output_path": intermediate_data_path,
        }
        feature_engineering.with_options(**pipeline_args)(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!\n")

        train_dataset_artifact = client.get_artifact_version(train_dataset_name)
        test_dataset_artifact = client.get_artifact_version(test_dataset_name)
        logger.info(
            "The latest feature engineering pipeline produced the following "
            "artifacts: \n\n1. Train Dataset - Name: %s, "
            "Version Name: %s \n2. Test Dataset: "
            "Name: %s, Version Name: %s",
            train_dataset_name,
            train_dataset_artifact.version,
            test_dataset_name,
            test_dataset_artifact.version,
        )

    # Execute Train Pipeline
    if train_pipeline:
        run_args_train = {"dataset_path": primary_data_path}

        # If train_dataset_version_name is specified, use versioned artifacts
        if train_dataset_version_name or test_dataset_version_name:
            # However, both train and test dataset versions must be specified
            assert train_dataset_version_name is not None
            assert test_dataset_version_name is not None
            train_dataset_artifact_version = client.get_artifact_version(
                train_dataset_name, train_dataset_version_name
            )
            # If train dataset is specified, test dataset must be specified
            test_dataset_artifact_version = client.get_artifact_version(
                test_dataset_name, test_dataset_version_name
            )
            # Use versioned artifacts
            run_args_train["train_dataset_id"] = train_dataset_artifact_version.id
            run_args_train["test_dataset_id"] = test_dataset_artifact_version.id

        # Run the SGD pipeline
        pipeline_args = {}
        if not use_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = Path(config_folder, "training_sgd.yaml")
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with SGD finished successfully!\n\n")

        # Run the RF pipeline
        pipeline_args = {}
        if not use_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = Path(config_folder, "training_rf.yaml")
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with RF finished successfully!\n\n")

    if inference_pipeline:
        run_args_inference = {}
        pipeline_args = {"enable_cache": False}
        pipeline_args["config_path"] = Path(config_folder, "inference.yaml")

        # Configure the pipeline
        inference_configured = inference.with_options(**pipeline_args)

        # Fetch the production model
        with pipeline_args["config_path"].open("r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        zenml_model = client.get_model_version(
            config["model"]["name"], config["model"]["version"]
        )
        preprocess_pipeline_artifact = zenml_model.get_artifact("preprocess_pipeline")

        # Use the metadata of feature engineering pipeline artifact
        #  to get the random state and target column
        random_state = preprocess_pipeline_artifact.run_metadata["random_state"]
        target = preprocess_pipeline_artifact.run_metadata["target"]
        run_args_inference["random_state"] = random_state
        run_args_inference["target"] = target

        # Run the pipeline
        inference_configured(**run_args_inference)
        logger.info("Inference pipeline finished successfully!")


@click.command()
def backend():
    # Change to frontend directory and run npm run dev
    cmd = "uvicorn quick_ds.apps.api.main:app --host 0.0.0.0 --port 5086 "
    process = Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, cwd=str(FRONTEND_DIR)
    )
    process.wait()


# Add commands to the CLI group
cli.add_command(zenml)
cli.add_command(backend)

if __name__ == "__main__":
    cli()
