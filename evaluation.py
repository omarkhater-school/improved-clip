import torch
import time
import utils
import torch.nn.functional as F
import torch.distributed as dist
import datetime
import numpy as np
import json, os
from zero_shot_helpers import zeroshot_transfer

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, output_hidden_states=False)  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds,dim=0)
    
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embed = F.normalize(image_embed, dim=-1)      
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds.to(device) @ text_embeds.to(device).t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_i2t[start+i, topk_idx] = topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_t2i[start+i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Model Retrieval Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result

def evaluate_model(val_loader, model, tokenizer, args, zeroshot_dataloader=None):
    """
    Evaluate the model on validation or test datasets and optionally perform zero-shot evaluation.
    """
    start_time = time.time()
    model.eval()
    device = args.device

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    print("***\nStarting evaluation\n***")

    # Evaluate on validation set
    score_val_i2t, score_val_t2i = evaluation(
        model_without_ddp,
        val_loader,
        tokenizer,
        device,
        args
    )

    # Compute metrics for validation
    val_result = itm_eval(
        score_val_i2t,
        score_val_t2i,
        val_loader.dataset.txt2img,
        val_loader.dataset.img2txt
    )
    print("Validation results:", val_result)

    # Optional zero-shot evaluation
    if zeroshot_dataloader:
        print("starting zeroshot transfer...")
        zeroshot_results = zeroshot_transfer(
            model_without_ddp,
            zeroshot_dataloader,
            args.zs_dataset,
            tokenizer,
            device
        )
        print("Zero-shot results:", zeroshot_results)
    else:
        zeroshot_results = None

    # Logging evaluation results
    if utils.is_main_process():
        log_stats = {
            **{f'val_{k}': v for k, v in val_result.items()},
            'epoch': args.start_epoch,  # Or specify evaluation-specific identifier
            'data': 'validation',
        }

        if zeroshot_results:
            log_stats.update({f'zeroshot_{k}': v for k, v in zeroshot_results.items()})

        # Save log to file
        log_file = os.path.join(args.output_dir, "eval_log.txt")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        print(f"Evaluation results logged to {log_file}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Overall Evaluation time {}'.format(total_time_str)) 
    return val_result, zeroshot_results

def get_objective_value(val_result, zeroshot_results = None):
    if zeroshot_results:
        zero_shot_score = zeroshot_results.get("zeroshot_top1")
    else:
        zero_shot_score = 0
    txt_r1 = val_result.get("txt_r1")
    img_r1 = val_result.get("img_r1")
    score = (zero_shot_score + txt_r1 + img_r1) / 3
    return score