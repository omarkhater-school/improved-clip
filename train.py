import utils
import torch
import os
import time
from evaluation import evaluation, itm_eval
from zero_shot_helpers import zeroshot_transfer
import pickle
import json
import torch.distributed as dist
import datetime
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
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(image, text, idx, text_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)   
        idx = idx.to(device, non_blocking=True)
        text_idx = text_idx.to(device, non_blocking=True)   
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
        
        # set learning rate for temperature network
        optimizer.param_groups[2]["lr"] = optimizer.param_groups[0]["lr"] / 10.0

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
        
        metric_logger.update(loss_ita=loss_ita.item())

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

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_temp_net=optimizer.param_groups[2]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and scheduler is not None: 
            scheduler.step(i//step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


#%% 
def train_model(train_loader, 
                val_loader,
                zeroshot_dataloader, 
                model,
                optimizer,
                tokenizer,
                lr_scheduler, 
                args
                ):
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.use_amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    best_epoch = 0
    max_epoch = args.epochs
    warmup_steps = args.warmup_epochs
    start_time = time.time()    

    for epoch in range(0, max_epoch):
        if not args.evaluate:

            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
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
                args)
            
        if args.evaluate:
            score_val_i2t_coco, score_val_t2i_coco = evaluation(
                model_without_ddp, 
                val_loader, 
                tokenizer, 
                args.device, 
                args
                )

        if utils.is_main_process():  

            if args.evaluate:
                val_result_coco = itm_eval(
                    score_val_i2t_coco, 
                    score_val_t2i_coco, 
                    val_loader.dataset.txt2img, 
                    val_loader.dataset.img2txt
                    )  
                print("coco val:", val_result_coco)


                if args.zs_dataset:
                    zeroshot_results = zeroshot_transfer(
                        model_without_ddp, 
                        zeroshot_dataloader, 
                        args.zs_dataset, 
                        tokenizer, 
                        args.device
                        )
                    print("zeroshot:", zeroshot_results)
                else:
                    zeroshot_results = None

            # save tau for visualization
            if not args.evaluate and args.store_tau and (epoch+1)%10==0:
                print("saving tau...")
                tau_image = model_without_ddp.criterion.tau_I.clone().cpu().numpy()
                tau_text = model_without_ddp.criterion.tau_T.clone().cpu().numpy()

                with open(os.path.join(args.output_dir, "tau_"+str(epoch)+".pkl"), "wb") as f:
                    pickle.dump({"tau_image":tau_image, "tau_text":tau_text}, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result_coco.items()},                 
                             'epoch': epoch,
                             'data': 'coco',
                            }
                with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")    

                if zeroshot_results:
                    with open(os.path.join(args.output_dir, f"zeroshot_{args.zs_dataset}_log.txt"), "a") as f:
                        f.write(json.dumps(zeroshot_results) + "\n")

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            #  **{f'val_{k}': v for k, v in val_result_coco.items()},                
                             'epoch': epoch,
                             'data': 'coco',
                            }
                with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_obj = {
                    'model': model_without_ddp.state_dict()
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch+1)+'.pth'))
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "coco_log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)  