import logging
from contextlib import suppress

from tqdm import tqdm
import torch
import torch.nn.functional as F
from clip import tokenize
from tf_yarn.pytorch.model_ckpt import _unwrap_model

from clip_on_yarn.validation.imagenet_zeroshot_data import (
    imagenet_classnames, openai_imagenet_template
)


logger = logging.getLogger()


def zero_shot_classifier(model, classes, templates, device):
    with torch.no_grad():
        zeroshot_classifier = []
        for classname in tqdm(classes, total=len(classes)):
            texts = [template(classname) for template in templates] 
            texts = tokenize(texts).to(device) 
            class_embeddings = _unwrap_model(model).encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_classifier.append(class_embedding)
     # stack along second dimension to avoid transpose when matrix multiplication
    return torch.stack(zeroshot_classifier, dim=1).to(device)


def accuracy(logits, target, topk=(1,)):
    # Indices of K largest logits (shape: (K, batch size))
    top_classes = logits.topk(max(topk), 1, True, True)[1].t()
    # True if the indice matches the target else False (shape: (K, batch size))
    correct = top_classes.eq(target.view(1, -1).expand_as(top_classes))
    # Flatten, convert to 1.0 or 0.0 and sum values
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(model, classifier, dataloader, device, precision, n_steps=None):
    autocast = torch.cuda.amp.autocast if precision == 'amp' else suppress
    with torch.no_grad():
        top1, top5, top10, n = 0., 0., 0., 0.
        for i, (images, target) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                image_features = _unwrap_model(model).encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
            top1 += acc1
            top5 += acc5
            top10 += acc10
            n += images.size(0)
            if n_steps and i >= n_steps:
                break

    top1 = (top1 / n)
    top5 = (top5 / n)
    top10 = (top10 / n)
    return {
        'zeroshot-val-top1': top1,
        'zeroshot-val-top5': top5,
        'zeroshot-val-top10': top10
    }


def zero_shot_eval(
    model, dataloader, device, precision,
    classes=imagenet_classnames, templates=openai_imagenet_template, n_steps=None
):
    logger.info('Building zero shot classifier')
    classifier = zero_shot_classifier(model, classes, templates, device)
    return evaluate(model, classifier, dataloader, device, precision, n_steps)
