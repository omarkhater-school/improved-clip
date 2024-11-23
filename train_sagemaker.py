import argparse
from sagemaker.pytorch import PyTorch
from config import role, region
import os, json
from config import aws_access_key_id, aws_secret_access_key
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

def parse_args():
    parser = argparse.ArgumentParser(description="Run SageMaker training job with CLIP.")
    # Accept all CLI arguments from the original script dynamically
    parser.add_argument("--job_name", required=True, help="SageMaker training job name.")
    parser.add_argument("--entry_point", default="main.py", help="Path to the main training script.")
    parser.add_argument("--instance_type", default="ml.c4.2xlarge", help="Instance type for training.")
    parser.add_argument("--source_dir", default=".", help="Directory containing training code.")
    parser.add_argument("--config_file", required=True, help="Path to JSON configuration file for the hyper-parameters.")
    parser.add_argument("--use_spot", action="store_true", help="Use spot instances for cost savings.")
    parser.add_argument("--max_wait", type=int, default=3600, help="Maximum wait time (in seconds) for spot instances.")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config_file, "r") as f:
        hyperparameters = json.load(f)
    
    # Spot instance configuration
    spot_config = {
        "use_spot_instances": args.use_spot,
        "max_wait": args.max_wait if args.use_spot else None,
    }
        
    estimator = PyTorch(
        entry_point=args.entry_point,
        source_dir=args.source_dir,
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.1",
        py_version="py310",
        output_path="s3://competitions23/CSCE636_DL_project/outputs",
        base_job_name=args.job_name,
        hyperparameters=hyperparameters,
        region_name=region,
        max_run = 5 * 3600, 
        environment={
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
    },
        metric_definitions=[ 
    {"Name": "LearningRate", "Regex": "lr=([0-9\\.]+)"},  # Learning rate metric
    {"Name": "Iteration_loss", "Regex": "Iteration [0-9]+, Epoch [0-9]+, Iteration Loss: ([0-9\\.]+)"},  # Iteration loss
    {"Name": "AvgImageTau", "Regex": "Average Image Tau: ([0-9\\.]+)"},  # Average Image Tau per epoch
    {"Name": "AvgTextTau", "Regex": "Average Text Tau: ([0-9\\.]+)"},  # Average Text Tau per epoch
    {"Name": "GradTauImage", "Regex": "Average Grad Tau Image: ([0-9\\.]+)"},  # Gradient Tau Image per epoch
    {"Name": "GradTauText", "Regex": "Average Grad Tau Text: ([0-9\\.]+)"},  # Gradient Tau Text per epoch
    {"Name": "AvgEpochLoss", "Regex": "Average Epoch Loss: ([0-9\\.]+)"},  # Epoch loss
    {"Name": "ValidationEpoch", "Regex": "Validation Epoch: ([0-9\\.]+)"},  # Validation epoch
    {"Name": "ObjectiveValue", "Regex": "objective value: ([0-9\\.]+)"},  # Validation objective value
    {"Name": "ValidationTxtR1", "Regex": "Validation txt_r1: ([0-9\\.]+)"},  # Validation Text R1 score
    {"Name": "ValidationImgR1", "Regex": "Validation img_r1: ([0-9\\.]+)"},  # Validation Image R1 score
    {"Name": "ValidationZS1", "Regex": "Validation zeroshot_top1: ([0-9\\.]+)"}  # Validation Zero-shot Top-1 accuracy
], 
        **spot_config
    )

    estimator.fit(
        {
            "train": f"s3://competitions23/CSCE636_DL_project/datasets"
        }, 
        wait= False
    )

if __name__ == "__main__":
    main()
