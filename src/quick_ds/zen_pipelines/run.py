# src/zen_pipelines/run_poc.py
from zenml import pipeline, step
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path


@step
def run_petrophysics_pipeline():
    # Bootstrap Kedro inside a ZenML step
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        # Runs your Kedro 'primary' pipeline
        output = session.run(pipeline_name="primary")
    return output


@pipeline
def poc_deployment_pipeline():
    run_petrophysics_pipeline()


if __name__ == "__main__":
    poc_deployment_pipeline()
