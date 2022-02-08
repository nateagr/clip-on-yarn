import io
from clip.model import convert_weights, CLIP
from clip.clip import tokenize
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            normalize,
        ])


def _convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def load_model(precision):
    vit_b_32_config = {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 32,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12
    }
    model = CLIP(**vit_b_32_config)
    if precision == "amp" or precision == "fp32":
        _convert_models_to_fp32(model)
    elif precision == "fp16":
        convert_weights(model)
    return model


def preprocessing(n_px: int, is_train: bool):
    preprocess_img = transform(n_px, is_train)
    def _preprocess_fn(img_text):
        image, text = img_text
        image_tensor = preprocess_img(Image.open(io.BytesIO(image)))
        text_tensor = tokenize([text], truncate=True)[0]
        return image_tensor, text_tensor
    return _preprocess_fn
