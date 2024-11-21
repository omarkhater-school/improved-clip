from torchvision import transforms, datasets
import torch
from zeroshot_transfer.classes import CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES
import torch.nn.functional as F
"""
    zero-shot transfer
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/evaluate.py#L42
"""

def create_zeroshot_dataloader(dataset_name, data_folder, image_size):
    assert dataset_name in ['cifar10', 'cifar100', 'imagenet']

    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_folder, download=False, train=False, transform=val_transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_folder, download=False, train=False, transform=val_transform)
    else:
        dataset = datasets.ImageFolder(root=data_folder, transform=val_transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                              num_workers=2, pin_memory=True)

    data_loader.num_samples = len(dataset)

    return data_loader



@torch.no_grad()
def zeroshot_transfer(model, data_loader, dataset_name, tokenizer, device):
    model.eval()

    if dataset_name == 'cifar10':
        config = CIFAR10_CLASSES
    elif dataset_name == 'cifar100':
        config = CIFAR100_CLASSES
    elif dataset_name == 'imagenet':
        config = IMAGENET_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    classes, templates = config["classes"], config["templates"]

    text_embeddings = []
    for c in classes:
        texts = [template(c) for template in templates]
        text_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_outputs = model.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask, output_hidden_states=False)  
        text_embeds = F.normalize(model.text_proj(text_outputs.last_hidden_state[:,0,:]), dim=-1)
        text_embed = text_embeds.mean(dim=0)
        text_embed /= text_embed.norm()
        text_embeddings.append(text_embed)

    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    topk = [1, 3, 5, 10]
    correct = {k: 0 for k in topk}

    for image, label in data_loader:
        image, label = image.to(device), label.to(device)
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embedding = F.normalize(image_embed, dim=-1)

        logits = image_embedding @ text_embeddings
        ranks = logits.topk(max(topk), 1)[1].T
        predictions = ranks == label

        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    results = {f"zeroshot_top{k}": 100.0 * correct[k] / data_loader.num_samples for k in topk}

    return results