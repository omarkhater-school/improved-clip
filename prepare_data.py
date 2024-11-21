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