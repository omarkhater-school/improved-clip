import utils
import torch
import os
import time
from evaluation import evaluate_model, get_objective_value
import pickle
import torch.distributed as dist
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
#%% 
def train_one_epoch(
        model, 
        data_loader, 
        optimizer, 
        tokenizer, 
        epoch, 
        max_epoch, 
        warmup_steps, 
        device, 
        scheduler, 
        grad_scaler, 
        args
        ):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_temp_net', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_image_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_text_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('cur_eta', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_image', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_text', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_I', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_T', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('v', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('lamda', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_image_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_text_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    step_size = args.step_size_per_epoch
    warmup_iterations = warmup_steps*step_size  
    progress_bar = tqdm(
        data_loader, 
        desc=header, 
        total=len(data_loader), 
        leave=True,
        position=0  # Ensure it overwrites correctly
    )
    for i,(image, text, idx, text_idx) in enumerate(progress_bar):
        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)   
        idx = idx.to(device, non_blocking=True)
        text_idx = text_idx.to(device, non_blocking=True)   
        text_input = tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=30, 
            return_tensors="pt"
        ).to(device)
        
        # set learning rate for temperature network
        optimizer.param_groups[2]["lr"] = optimizer.param_groups[0]["lr"] / 10.0

        # Compute loss and update model
        if grad_scaler is None:
            loss_ita, info_dict = model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
            loss_ita.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                loss_ita, info_dict = model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
            grad_scaler.scale(loss_ita).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        
        

        if args.ita_type in ['sogclr_dro', 'isogclr_new']:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)
        elif args.ita_type == 'isogclr_new_v2':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=info_dict['v'])
            metric_logger.update(lamda=info_dict['lamda'])
        elif args.ita_type == 'sogclr':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=info_dict['lamda'])
        else:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)

        
        metric_logger.update(loss_ita=loss_ita.item())   
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_temp_net=optimizer.param_groups[2]["lr"])
        if i % args.print_freq_per_epoch == 0:
            progress_bar.set_postfix(
                loss_ita=f"{loss_ita.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and scheduler is not None: 
            scheduler.step(i//step_size)

        print(f"Iteration {i + 1}, Epoch {epoch + 1}, Iteration Loss: {loss_ita.item():.4f}")
    progress_bar.close()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger.global_avg())
    train_stats =  {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

    print(f"""
          Epoch {epoch + 1}
          Average Epoch Loss: {metric_logger.meters['loss_ita'].global_avg}
          Average Image Tau: {metric_logger.meters['avg_image_tau'].global_avg}
          Average Text Tau: {metric_logger.meters['avg_text_tau'].global_avg}, 
          Average Grad Tau Image: {metric_logger.meters['grad_tau_image'].global_avg}
          Average Grad Tau Text: {metric_logger.meters['grad_tau_text'].global_avg}
        """)

    return model, train_stats


#%% 
def train_model(
        train_loader, 
        model, 
        optimizer, 
        tokenizer, 
        lr_scheduler, 
        args, 
        val_loader, 
        zeroshot_dataloader = None
                ):
    """
    Train the model with support for resume learning and saving checkpoints and tau.
    """
    # Wrap the model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up gradient scaler for mixed precision training
    grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Initialize training parameters
    start_epoch = args.start_epoch
    max_epoch = args.epochs
    warmup_steps = args.warmup_epochs
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        # Update sampler for distributed training
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        model, _ = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            tokenizer, 
            epoch, 
            max_epoch, 
            warmup_steps, 
            args.device, 
            lr_scheduler, 
            grad_scaler, 
            args
        )

        # Evaluate the model at specified intervals
        if (epoch + 1) % args.val_frequency == 0:
            print(f"\n*** Evaluation at Epoch {epoch + 1} ***\n")
            val_result, zeroshot_results = evaluate_model(val_loader, model_without_ddp, tokenizer, args, zeroshot_dataloader)
            objective_value = get_objective_value(val_result, zeroshot_results)

            print(
                f"""
                  Validation Epoch: {epoch + 1}
                  Validation txt_r1: {val_result.get("txt_r1")}
                  Validation img_r1: {val_result.get("img_r1")}
                  Validation zeroshot_top1: {zeroshot_results.get("zeroshot_top1")}
                  objective value: {objective_value}
                """
                )
        # Save tau values every 10 epochs (if requested)
        if args.store_tau and (epoch + 1) % 10 == 0:
            print("Saving tau values...")
            tau_image = model_without_ddp.criterion.tau_I.clone().cpu().numpy()
            tau_text = model_without_ddp.criterion.tau_T.clone().cpu().numpy()
            with open(os.path.join(args.output_dir, f"tau_{epoch + 1}.pkl"), "wb") as f:
                pickle.dump({"tau_image": tau_image, "tau_text": tau_text}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save checkpoint after every epoch
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': vars(args),
            'epoch': epoch,
        }
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch + 1}.pth')
        torch.save(save_obj, checkpoint_path)

        # Step the learning rate scheduler
        lr_scheduler.step(epoch + warmup_steps + 1)

        if args.distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    # Log total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    
    return model_without_ddp

def load_model_from_checkpoint(model, args):
    import re
    if args.resume_learning:
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu') 
            state_dict = checkpoint['model']             
            model.load_state_dict(state_dict, strict=False)  
            print('load checkpoint from %s' % args.checkpoint)
            match = re.search(r'checkpoint_(\d+)\.pth', args.checkpoint)
            if match:
                start_epoch = int(match.group(1))
            else:
                start_epoch = 0
                print("No checkpoint found. Starting from 0")
        except Exception as e:
            print(f"Failed to load checkpoint due to \n{e}")
            start_epoch = 0
            model = model
    else:
        start_epoch = 0
        model = model
    return model, start_epoch

def extract_and_save_sample_tau(train_loader, model, tokenizer, args):

    if args.check_samples_tau:
        image_tau_array = []
        text_tau_array = []

        model.eval() 
    
        with torch.no_grad():
            for image, text, idx, text_idx in tqdm(train_loader):
                image = image.to(args.device)
                text = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(args.device)

                image_feat = F.normalize(model.vision_proj(model.visual_encoder(image)), dim=-1)
                text_output = model.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
                text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
            
                tau_image = model.criterion.image_temp_gen(image_feat).cpu().squeeze().numpy()
                tau_text = model.criterion.text_temp_gen(text_feat).cpu().squeeze().numpy()

                image_tau_array.append(tau_image)
                text_tau_array.append(tau_text)

            image_tau_array = np.concatenate(image_tau_array) 
            text_tau_array = np.concatenate(text_tau_array)

        with open(os.path.join(args.output_dir, "tau.pkl"), "wb") as f:
            pickle.dump({"tau_image":image_tau_array, "tau_text":text_tau_array}, f, protocol=pickle.HIGHEST_PROTOCOL)

        assert 0