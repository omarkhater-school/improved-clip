from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role='your-sagemaker-role',
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='1.12.0',
    py_version='py38',
    hyperparameters={
        "batch_size_train": 128,
        "epochs": 30,
        "image_res": 256,
    }
)
