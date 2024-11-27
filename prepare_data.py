from pathlib import Path
from dataset import create_train_dataset, create_val_dataset, create_sampler, create_train_loader, create_val_loader
import torchvision
import shutil
from tqdm import tqdm
import os
from torch.utils.data import Subset
import utils

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


def prepare_output_path(args):
    """
    Prepare the output folder based on the environment.
    - If running in SageMaker, use SageMaker's expected output directory.
    - Otherwise, use a local path.
    """
    if "SM_OUTPUT_DATA_DIR" in os.environ:  # Check if running in SageMaker
        root_output_path = os.environ["SM_OUTPUT_DATA_DIR"]
    else:  # Local environment
        root_output_path = args.output_dir

    # Ensure the directory exists
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)

    args.output_dir = root_output_path
    return args


def manage_paths_and_environment(args):
    """
    Centralized method to manage paths and environment:
    - Prepare data paths (including S3 downloads).
    - Prepare output path based on the environment.
    """
    args = prepare_output_path(args)
    return args