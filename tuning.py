import argparse
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter, IntegerParameter
from sagemaker.pytorch import PyTorch
from config import role, region
import os, json
from config import aws_access_key_id, aws_secret_access_key
import sagemaker
import boto3
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

METRICS = [ 
            {"Name": "LearningRate", "Regex": "lr=([0-9\\.]+)"},
            {"Name": "Iteration_loss", "Regex": "loss_ita=([0-9\\.]+)"},
            {"Name": "AvgImageTau", "Regex": "Average Image Tau: ([0-9\\.]+)"},
            {"Name": "AvgTextTau", "Regex": "Average Text Tau: ([0-9\\.]+)"},
            {"Name": "GradTauImage", "Regex": "Average Grad Tau Image: ([0-9\\.]+)"},
            {"Name": "GradTauText", "Regex": "Average Grad Tau Text: ([0-9\\.]+)"},
            {"Name": "AvgEpochLoss", "Regex": "Average Epoch Loss: ([0-9\\.]+)"},
            {"Name": "ValidationEpoch", "Regex": "Validation Epoch: ([0-9\\.]+)"},
            {"Name": "ObjectiveValue", "Regex": "objective value: ([0-9\\.]+)"},
            {"Name": "BestObjectiveValue", "Regex": "Best Objective value ([0-9\\.]+)"},
            {"Name": "ValidationTxtR1", "Regex": "Validation txt_r1: ([0-9\\.]+)"},
            {"Name": "ValidationImgR1", "Regex": "Validation img_r1: ([0-9\\.]+)"},
            {"Name": "ValidationZS1", "Regex": "Validation zeroshot_top1: ([0-9\\.]+)"},
            {"Name": "MovingAvgObjectiveValue", "Regex": "Moving Avg Objective value: ([0-9\\.]+)"}
        ]

def parse_args():
    parser = argparse.ArgumentParser(description="Run SageMaker training job with CLIP.")
    parser.add_argument("--job_name", required=True, help="SageMaker tuning job name.")
    parser.add_argument("--entry_point", default="main.py", help="Path to the main training script.")
    parser.add_argument("--instance_type", default="ml.c4.2xlarge", help="Instance type for training.")
    parser.add_argument("--source_dir", default=".", help="Directory containing training code.")
    parser.add_argument("--config_file", required=True, help="Path to JSON configuration file for the hyper-parameters.")
    parser.add_argument("--use_spot", action="store_true", help="Use spot instances for cost savings.")
    parser.add_argument("--max_wait", type=int, default=3600, help="Maximum wait time (in seconds) for spot instances.")
    return parser.parse_args()


def create_parameter(config):
    """
    Convert a parameter configuration into SageMaker-compatible hyperparameter ranges.
    """
    param_type = config["type"]
    if param_type == "Continuous":
        return ContinuousParameter(
            config["range"][0], 
            config["range"][1], 
            scaling_type=config.get("scale", "Linear")
            )
    elif param_type == "Categorical":
        return CategoricalParameter(config["values"])
    elif param_type == "Integer":
        return IntegerParameter(config["range"][0], config["range"][1])
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")
    
def get_hyperparameter_ranges(config, loss_function):
    """
    Extract hyperparameter ranges based on the loss function and optimizer.
    """
    hyperparameter_ranges = {}
    print(config.keys())
    # Add common optimizer ranges
    for param, param_config in config["Optimizer"].items():
        hyperparameter_ranges[param] = create_parameter(param_config)

    # Add scheduler ranges
    for param, param_config in config['scheduler'].items():
        hyperparameter_ranges[param] = create_parameter(param_config)

    # Add common loss-specific ranges
    for param, param_config in config["loss_specific"]["common"].items():
        hyperparameter_ranges[param] = create_parameter(param_config)

    # Add ranges specific to the selected loss function
    if loss_function in config["loss_specific"]:
        for param, param_config in config["loss_specific"][loss_function].items():
            hyperparameter_ranges[param] = create_parameter(param_config)

    # Add model-specific ranges
    for param, param_config in config["model_specific"].items():
        hyperparameter_ranges[param] = create_parameter(param_config)

    return hyperparameter_ranges


def create_estimator(args, hyperparameters):
    spot_config = {
        "use_spot_instances": args.use_spot,
        "max_wait": args.max_wait if args.use_spot else None,
    }
    
    # Create a boto3 session and sagemaker session with the specified region
    
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    estimator = PyTorch(
        entry_point=args.entry_point,
        source_dir=args.source_dir,
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.1",
        py_version="py310",
        output_path=f"s3://competitions23/CSCE636_DL_project/outputs/{args.job_name}",
        base_job_name=args.job_name,
        hyperparameters=hyperparameters,
        max_run=8 * 3600,
        metric_definitions= METRICS,
        sagemaker_session=sagemaker_session,
        **spot_config
    )
    return estimator


def run_tuning_job(
        args, 
        train_frac, 
        max_jobs, 
        max_parallel_jobs, 
        epochs,
        loss_function, 
        optimizer
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

    estimator = create_estimator(args, hyperparameters)

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="BestObjectiveValue",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions= METRICS, 
        strategy='Bayesian', 
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        base_tuning_job_name = args.job_name
    )

    tuner.fit({
        "train": f"s3://competitions23/CSCE636_DL_project/datasets"
    }, 
    wait= False)

def main():
    args = parse_args()
    combinations = [
        ("isogclr_new", "radam"),
        # ("isogclr_new", "adamw"),
        # ("sogclr", "adamw"), 
        # ("sogclr", "radam")

    ]
    for loss_function, optimizer in combinations:
        run_tuning_job(
            args=args,
            train_frac=1,
            max_jobs=30,
            max_parallel_jobs=3,
            epochs=30,
            loss_function=loss_function,
            optimizer=optimizer
        )

if __name__ == "__main__":
    main()
