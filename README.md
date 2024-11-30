# **Optimizing Global Contrastive Loss for Multimodal Learning**

This repository explores and optimizes global contrastive loss functions, focusing on enhancing image-text representation learning. The project was developed as part of the **CSCE 636 Deep Learning course (Fall 24) at Texas A&M University**.

---
## Benchmark Comparison


| **Method**        | **MSCOCO TR@1** | **MSCOCO IR@1** | **ImageNet ACC@1** | **Average** |
|--------------------|-----------------|------------------|--------------------|-------------|
| CLIP (Benchmark)   | 12.00           | 9.32             | 21.35              | 14.22       |
| SogCLR (Provided codebase)          | 14.38           | 10.73            | 24.54              | 16.55       |
| iSogCLR\_New (Ours)| **15**       | **11.52**        | **28.54**          | **18.33**   |


## **Repository Structure**

### **Key Folders**
- **`models/`**: Image and text encoders (ResNet-50 and DistilBERT) with modular loss functions.  
- **`notebooks/`**: Jupyter notebooks for result analysis and experimentation.  
- **`optim/`**: Custom optimizers including AdamW, RAdam, and NovoGrad.  
- **`scheduler/`**: Learning rate schedulers for warmup, cooldown, and decay.  
- **`zeroshot_transfer/`**: Evaluation scripts for zero-shot classification.  
- **`documentation/`**: Contains the **project report** detailing the methodology, experiments, and results.
- **`final_model/`**: Contains the final trained model for inference. 

---

## **Key Improvements**
This repository extends the original provided codebase with several enhancements:
1. **AWS SageMaker Integration**: Enables seamless training of models in distributed environments using SageMaker.
2. **Modularized Code**: Refactored for easy integration of new loss functions, optimizers, and datasets.
3. **Advanced Hyperparameter Tuning**: Incorporates Bayesian optimization for tuning critical parameters such as learning rates, temperature, and regularization.
4. **Robust Evaluation Pipeline**: Enhanced evaluation metrics and dataset handling for retrieval and classification tasks.

---

## **Getting Started**
1. **Prepare datasets**: Ensure the dataset folder structure matches:

- **`datasets/`**: Organized datasets for training and validation:  
  - **`cc3m_subset_100k/`**: Training subset of Conceptual Captions 3M.  
  - **`clip_train/`**: Metadata for training and validation datasets.  
  - **`mscoco_val/`**: MSCOCO validation data (image-text retrieval).  
  - **`imagenet/`**: ImageNet validation data (zero-shot classification).
    
2. **Install dependencies**:
   
  ```bash
    pip install -r requirements.txt
  ``` 
3. [Optional] If you want to use the sagemaker training, and tuning, make sure to create `config.py` with the following: 
  ```python
  role = <Your Sagemaker role>
  region = <Your AWS region>
  aws_access_key_id = <Your acess key id>
  aws_secret_access_key = <Your secret acess key>
  ```
4. Test Run the main script
   
  ```bash
  python main.py \
    --data_path "./datasets" \
    --ann_path "datasets/clip_train" \
    --zs_datafolder "datasets/imagenet/val" \
    --train_file cc3m_train_subset.json \
    --train_image_root cc3m_subset_100k \
    --output_dir "./test_output" \
    --loss_function isogclr_new \
    --optimizer fusedadam \
    --tau_init 0.01 \
    --sogclr_gamma 0.8 \
    --eta_init 0.03 --sched cosine \
    --device cuda \
    --val_frequency 5 \
    --epochs 1
   ```
4. Test Create Sagemaker training job
   
   ```bash
   python train_sagemaker.py \
    --entry_point main.py \
    --source_dir . \
    --instance_type ml.g5.4xlarge\
    --use_spot \
    --max_wait 36000 \
    --config_file ./config.json \
    --job_name "Test-improve-clip"
   ```
6. Test Create Sagemaker tuning job
   
   ```bash
   python tuning.py \
    --entry_point main.py \
    --source_dir . \
    --instance_type ml.g5.4xlarge\
    --use_spot \
    --max_wait 36000 \
    --config_file ./config_phase3.json \
    --job_name "improved-clip-phase3"
   ```
7. [Optional] Continue Tune existing finished tuning job
   
   ```bash
   python warm_start_tuning.py \
    --job_name improved-clip-phase3-extended \
    --entry_point main.py \
    --instance_type ml.g5.4xlarge\
    --source_dir . \
    --config_file phase3_extended.json \
    --max_wait 36000 \
    --previous_job_name improved-clip-phase3-241127-1727\
    --objective_metric_name BestObjectiveValue\
    --use_spot
   ```

## Known Bugs
---
- isogclr_new_v1 didn't work with existing code
- `isogclr_temp_net=1` break the training  

## Citation
---
If you use this work, please cite it as follows:
```
@misc{omarkhater2024improvedclip,
  author       = {Omar Khater},
  title        = {Improving CLIP Training with Bayesian Optimization},
  year         = {2024},
  url          = {[https://github.com/your-username/improved-clip](https://github.com/omarkhater-school/improved-clip)},
```
