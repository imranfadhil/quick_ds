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
BACKEND_DIR = Path(__file__).parent / "apps" / "api"


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
    python main.py --pipeline feature

  \b
  # Run the training pipeline
    python main.py --pipeline train

  \b
  # Run the inference pipeline
    python main.py --pipeline inference
  
  \b 
  # Run the training pipeline with versioned artifacts
    python main.py --pipeline train --train-dataset-version-name=1 --test-dataset-version-name=1

"""
)
@click.option(
    "--train-config-file",
    "-c",
    default="training_rfr.yaml",
    type=click.STRING,
    help="The name of the train config file.",
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
    "--pipeline",
    "-p",
    type=click.Choice(["feature", "train", "dnn_train", "inference", "all"]),
    default=None,
    help="Pipeline to run",
)
@click.option(
    "--use-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def zenml(
    train_config_file: str = "training_rfr.yaml",
    train_dataset_name: str = "dataset_trn",
    train_dataset_version_name: str | None = None,
    test_dataset_name: str = "dataset_tst",
    test_dataset_version_name: str | None = None,
    pipeline: str | None = None,
    use_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        train_config_file: The name of the train config file.
        train_dataset_name: The name of the train dataset produced by feature engineering.
        train_dataset_version_name: Version of the train dataset produced by feature engineering.
            If not specified, a new version will be created.
        test_dataset_name: The name of the test dataset produced by feature engineering.
        test_dataset_version_name: Version of the test dataset produced by feature engineering.
            If not specified, a new version will be created.
        pipeline: Pipeline to run.
        use_cache: If `True` cache will be used.
    """
    client = Client()

    source_config_folder = Path(__file__).resolve().parent / "configs"
    config_folder = Path.cwd() / "configs"

    if not config_folder.exists():
        logger.info("Configs folder not found in CWD. Copying default configs...")
        shutil.copytree(source_config_folder, config_folder)
        logger.info("Configs copied to %s", config_folder)

    # Execute Feature Engineering Pipeline
    if pipeline in ["feature", "all"]:
        pipeline_args = {}
        if not use_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = Path(config_folder, "feature_engineering.yaml")
        run_args_feature = {}
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
    if pipeline in ["train", "all"]:
        run_args_train = {}

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
        pipeline_args["config_path"] = Path(config_folder, train_config_file)
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info(
            "Training pipeline with %s finished successfully!\n\n", train_config_file
        )

    # Execute DNN Train Pipeline
    if pipeline in ["dnn_train", "all"]:
        from quick_ds.pipelines import dnn_training

        run_args_train = {}

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

        # Run the DNN pipeline
        pipeline_args = {}
        if not use_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = Path(config_folder, train_config_file)

        dnn_training.with_options(**pipeline_args)(**run_args_train)
        logger.info(
            "DNN Training pipeline with %s finished successfully!\n\n",
            train_config_file,
        )

    if pipeline in ["inference", "all"]:
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
    process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    process.wait()


@click.command()
def frontend():
    # Change to frontend directory and run npm run dev
    cmd = " "
    process = Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, cwd=str(FRONTEND_DIR)
    )
    process.wait()


# Add commands to the CLI group
cli.add_command(zenml)
cli.add_command(backend)
cli.add_command(frontend)

if __name__ == "__main__":
    cli()
