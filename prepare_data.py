from pathlib import Path
from dataset import create_train_dataset, create_val_dataset, create_sampler, create_train_loader, create_val_loader
import torchvision
import shutil
from tqdm import tqdm
import os
from torch.utils.data import Subset
import utils
import boto3

def prepare_data_loaders(args):
    train_dataset = create_train_dataset('re', 
                                         args)
    val_coco_dataset = create_val_dataset('re', 
                                          args, 
                                          args.val_coco_file, 
                                          args.coco_image_root
                                          )
    
    print("len of train_dataset:", len(train_dataset))
    print("len of validation dataset:", len(val_coco_dataset))


    if args.extract_data:
        idx_list = []
        data_dir = os.path.join(args.output_dir, '')
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        for idx in tqdm(idx_list):
            image, text, _, _ = train_dataset.__getitem__(idx, enable_transform=False)
            torchvision.utils.save_image(image, fp=os.path.join(data_dir, str(idx)+':'+text+'.png'))
            
        shutil.make_archive(data_dir, 'zip', data_dir)

        assert 0

    num_training = int(args.train_frac * len(train_dataset))
    train_dataset = Subset(train_dataset, list(range(num_training)))


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader = create_train_loader(train_dataset, samplers[0], args.batch_size_train, 2, None)
    val_coco_loader = create_val_loader([val_coco_dataset], samplers[1:2], 
                                        [args.batch_size_test], [8], [None])[0] 

    return  train_loader, val_coco_loader  


def download_from_s3(s3_path, local_path):
    """
    Download data from S3 to the specified local path.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    s3 = boto3.client('s3')
    bucket, key = s3_path[5:].split('/', 1)  # Split "s3://bucket/key"
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    s3.download_file(bucket, key, os.path.join(local_path, os.path.basename(key)))
    return

def prepare_output_path(args):
    """
    Prepare the output folder based on the environment.
    - If running in SageMaker, use SageMaker's expected output directory.
    - Otherwise, use a local path.
    """
    if "SM_OUTPUT_DATA_DIR" in os.environ:  # Check if running in SageMaker
        root_output_path = os.environ["SM_OUTPUT_DATA_DIR"]
    else:  # Local environment
        root_output_path = os.path.join(args.data_path, "outputs")

    # Ensure the directory exists
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)

    args.output_dir = root_output_path
    return args


def prepare_data_path(args):
    """
    Prepare the data for training:
    - Download from S3 if necessary.
    - Map paths for local usage.
    """
    local_data_path = "./datasets_local"
    local_ann_path = "./annotations_local"
    local_train_file = os.path.join(local_data_path,"cc3m_train_subset.json")
    local_train_image_root = os.path.join(local_data_path,"cc3m_subset_100k")
    
    # Download data_path if it is an S3 path
    if args.data_path.startswith("s3://"):
        print(f"Downloading data from {args.data_path} to {local_data_path}")
        download_from_s3(args.data_path, local_data_path)
    
    # Download ann_path if it is an S3 path
    if args.ann_path.startswith("s3://"):
        print(f"Downloading annotations from {args.ann_path} to {local_ann_path}")
        download_from_s3(args.ann_path, local_ann_path)

    # Ensure paths are updated to local equivalents
    args.data_path = local_data_path
    args.ann_path = local_ann_path
    args.train_file = local_train_file
    args.train_image_root = local_train_image_root

    return args

def manage_paths_and_environment(args):
    """
    Centralized method to manage paths and environment:
    - Prepare data paths (including S3 downloads).
    - Prepare output path based on the environment.
    """
    args = prepare_data_path(args)
    args = prepare_output_path(args)
    return args