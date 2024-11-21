import warnings
import argparse
from models.model_clip import CLIP
from transformers import AutoTokenizer, RobertaTokenizer
from scheduler import create_scheduler
from optim import create_optimizer
from evaluation import run_eval
import torch
from zero_shot_helpers import create_zeroshot_dataloader
from prepare_data import prepare_data_loaders
import utils
from utils import warn, set_random, set_path
from train import train_model


warnings.warn = warn
#%% 
def run_pipeline(args):
    if args.distributed:
        utils.init_distributed_mode(args)    
    else:
        args.gpu = 0
    
    device = torch.device(args.device)
    set_random(args.seed)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    #### Dataset #### 
    print("Creating retrieval dataset")
    train_loader, val_coco_loader = prepare_data_loaders(args)
    if args.text_encoder == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(args.text_encoder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    #### Zero-shot transfer ####
    if args.zs_dataset:
        zeroshot_dataloader = create_zeroshot_dataloader(
            dataset_name=args.zs_dataset, 
            data_folder=args.zs_datafolder, 
            image_size=args.image_res
            )
    else:
        zeroshot_dataloader = None

    #### Model #### 
    print("Creating model")
    model = CLIP(
        image_encoder=args.image_encoder, 
        text_encoder=args.text_encoder, 
        embed_dim=args.embed_dim, 
        init_model=args.init_model, 
        bsz=args.batch_size_train*args.world_size,
        world_size=args.world_size, 
        ita_type=args.ita_type, 
        sogclr_gamma=args.sogclr_gamma, 
        rho_I=args.rho_I, 
        rho_T=args.rho_T, 
        tau_init=args.tau_init,
        eta_init=args.eta_init, 
        beta_u=args.beta_u, 
        temp=args.temp, 
        learnable_temp=args.learnable_temp,
        vicreg_sim_coeff=args.vicreg_sim_coeff, 
        vicreg_std_coeff=args.vicreg_std_coeff, 
        personalized_tau=args.personalized_tau, 
        use_temp_net=args.isogclr_temp_net, 
        alpha=args.alpha, 
        distributed=args.distributed
                 )
    model = model.to(device)


    ## Evaluation
    run_eval(train_loader, model, tokenizer, args)

    ## Training
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    print("Start training")

    train_model(train_loader, 
                val_coco_loader,
                zeroshot_dataloader, 
                model,
                optimizer,
                tokenizer,
                lr_scheduler, 
                args
                )
               

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--data_path', default='./datasets')
    parser.add_argument('--ann_path', default='./clip_train')
    parser.add_argument('--train_file', default='downstream/cc3m_train_new.json')
    parser.add_argument('--train_image_root', default='cc3m')

    # model config
    parser.add_argument('--bert_config', default='configs/config_bert.json')
    parser.add_argument('--image_encoder', default='resnet50')
    parser.add_argument('--text_encoder', default='distilbert-base-uncased')
    parser.add_argument('--image_res', default=256, type=int)
    parser.add_argument('--vision_width', default=768, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)

    # optimizer and schedular
    parser.add_argument('--opt', default='adamW')
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_temp_net', default=1e-6, type=float)
    parser.add_argument('--wd_temp_net', default=1e-3, type=float,
                        help='weight decay for temperature network')
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup', default=True, type=bool)
    parser.add_argument('--warmup_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--decay_rate', default=1, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--cooldown_epochs', default=0, type=int)

    # training & test settings
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--init_model', action='store_true')
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--k_test', default=256, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--no-distributed', dest='distributed', action='store_false')

    # output path
    parser.add_argument('--output_dir', default='./output/clip_test')  

    # loss config
    parser.add_argument('--ita_type', required=True, choices=['clip', 'cyclip', 'vicreg', 'sogclr', 'sogclr_dro', 
                        'isogclr_new_v2', 'isogclr_new_v1', 'isogclr_new', 'onlineclr'])
    parser.add_argument('--vicreg_sim_coeff', default=25.0, type=float)
    parser.add_argument('--vicreg_std_coeff', default=25.0, type=float)
    parser.add_argument('--sogclr_gamma', default=0.8, type=float)
    parser.add_argument('--rho_I', default=8.0, type=float)
    parser.add_argument('--rho_T', default=8.0, type=float)
    parser.add_argument('--eta_init', default=0.001, type=float)
    parser.add_argument('--tau_init', default=0.01, type=float)
    parser.add_argument('--beta_u', default=0.9, type=float)
    parser.add_argument('--temp', default=0.01, type=float)
    parser.add_argument('--learnable_temp', action='store_true')
    parser.add_argument('--personalized_tau', action='store_true')
    parser.add_argument('--max_norm', default=1.0, type=float)
    parser.add_argument('--store_tau', action='store_true')
    parser.add_argument('--isogclr_temp_net', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float, help='for isogclr_denoise')

    parser.add_argument('--train_frac', 
                        help="fraction of data used for training",
                        default=1.0, 
                        type=float)
    parser.add_argument('--check_samples_tau', 
                        help = "check samples with high/low temperature values",
                        action='store_true')
    parser.add_argument('--extract_data', 
                        help = "extract data from the cc3m dataset", 
                        action='store_true')
    parser.add_argument('--zs_dataset', 
                        help = "zero-shot transfer", 
                        default="", 
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--zs_datafolder', default='./datasets', type=str)

    args = parser.parse_args()

    if args.check_samples_tau:
        args.evaluate = True
    args = set_path(args)

    run_pipeline(args)


