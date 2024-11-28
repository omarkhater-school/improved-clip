import argparse
from tuning import create_estimator, get_hyperparameter_ranges, METRICS
from sagemaker.tuner import HyperparameterTuner, WarmStartConfig, WarmStartTypes
import os
import json
from config import aws_access_key_id, aws_secret_access_key, role, region
import boto3
import sagemaker

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key


def parse_args():
    parser = argparse.ArgumentParser(description="Run warm-start SageMaker tuning job with CLIP.")
    parser.add_argument("--job_name", required=True, help="SageMaker tuning job name.")
    parser.add_argument("--entry_point", default="main.py", help="Path to the main training script.")
    parser.add_argument("--instance_type", default="ml.c4.2xlarge", help="Instance type for training.")
    parser.add_argument("--source_dir", default=".", help="Directory containing training code.")
    parser.add_argument("--config_file", required=True, help="Path to JSON configuration file for the hyper-parameters.")
    parser.add_argument("--previous_job_name", required=True, help="Previous tuning job to warm start from.")
    parser.add_argument("--use_spot", action="store_true", help="Use spot instances for cost savings.")
    parser.add_argument("--max_wait", type=int, default=3600, help="Maximum wait time (in seconds) for spot instances.")
    parser.add_argument("--objective_metric_name", default="BestObjectiveValue", help="Metric to optimize during tuning.")
    return parser.parse_args()


def run_warm_start_tuning_job(
    args, train_frac, max_jobs, max_parallel_jobs, epochs, loss_function, optimizer
):
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # Add static hyperparameters
    hyperparameters = config["non-tunable"]
    hyperparameters["epochs"] = epochs
    hyperparameters["train_frac"] = train_frac
    hyperparameters["loss_function"] = loss_function
    hyperparameters["optimizer"] = optimizer

    # Create hyperparameter ranges dynamically based on loss and optimizer
    hyperparameter_ranges = get_hyperparameter_ranges(config, loss_function)

    # Create estimator
    estimator = create_estimator(args, hyperparameters)

    # Warm-start configuration
    warm_start_config = WarmStartConfig(
        parents=[args.previous_job_name],
        warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
    )

    # Use the specified objective metric for tuning
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=args.objective_metric_name,
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=METRICS,
        strategy="Bayesian",
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        base_tuning_job_name=args.job_name,
        warm_start_config=warm_start_config,
        early_stopping_type="Auto"  # Enable early stopping
    )

    # Start tuning job
    tuner.fit({"train": f"s3://competitions23/CSCE636_DL_project/datasets"}, wait=False)


def main():
    args = parse_args()
    combinations = [
        ("isogclr_new", "radam"),
        # Uncomment more combinations as needed
        # ("isogclr_new", "adamw"),
        # ("sogclr", "adamw"),
        # ("sogclr", "radam"),
    ]
    for loss_function, optimizer in combinations:
        run_warm_start_tuning_job(
            args=args,
            train_frac=1,
            max_jobs=10,
            max_parallel_jobs=2,
            epochs=30,
            loss_function=loss_function,
            optimizer=optimizer,
        )


if __name__ == "__main__":
    main()
