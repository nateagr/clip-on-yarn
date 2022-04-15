import logging
from contextlib import suppress

from tqdm import tqdm
import torch
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
        for classname in classes:
            texts = [template(classname) for template in templates] 
            texts = tokenize(texts).to(device) 
            class_embeddings = _unwrap_model(model).encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_classifier.append(class_embedding)
     # stack along second dimension to avoid transpose when matrix multiplication
    return torch.stack(zeroshot_classifier, dim=1).to(device)


def accuracy(logits, target, topk=(1,)):
    top_classes = logits.topk(max(topk), 1, True, True)[1].t()
    correct = top_classes.eq(target.view(1, -1).expand_as(top_classes))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(model, classifier, dataloader, device, batch_size, precision):
    autocast = torch.cuda.amp.autocast if precision == 'amp' else suppress
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=batch_size):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                image_features = _unwrap_model(model).encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            break

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(
    model, dataloader, device, batch_size, precision,
    classes=imagenet_classnames, templates=openai_imagenet_template
):
    logger.info('Starting zero-shot evaluation')
    classifier = zero_shot_classifier(model, classes, templates, device)
    top1, top5 = evaluate(model, classifier, dataloader, device, batch_size, precision)
    logger.info('Finished zero-shot evaluation')
    return {
        'zeroshot-val-top1': top1,
        'zeroshot-val-top5': top5
    }
