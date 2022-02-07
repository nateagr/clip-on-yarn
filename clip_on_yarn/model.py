from clip.model import convert_weights, CLIP
import io

from clip.clip import _transform
from clip.clip import tokenize
from PIL import Image


preprocess_img = _transform(224)


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


def preprocessing(img_text):
    image, text = img_text
    image_tensor = preprocess_img(Image.open(io.BytesIO(image)))
    text_tensor = tokenize([text], truncate=True)[0]
    return image_tensor, text_tensor
