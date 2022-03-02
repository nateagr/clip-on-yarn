from PIL import Image
import io
from functools import partial
import clip
import webdataset as wds


def preprocess_dataset(
    row, caption_key, image_key, metadata_key, image_transform, enable_image, enable_text, enable_metadata
):
    output = {}
    if enable_image:
        image_data = row[image_key]
        image = Image.open(io.BytesIO(image_data))
        image_tensor = image_transform(image)
        output["image_filename"] = row["__key__"]
        output["image_tensor"] = image_tensor

    if enable_text:
        text = row[caption_key]
        caption = text.decode("utf-8")
        tokenized_text = clip.tokenize([caption], truncate=True)[0]
        output["text_tokens"] = tokenized_text
        output["text"] = caption

    if enable_metadata:
        metadata_file = row[metadata_key]
        metadata = metadata_file.decode("utf-8")
        output["metadata"] = metadata
    return output


def filter_row(
    row, caption_key, image_key, metadata_key, enable_text, enable_image, enable_metadata
):
    return (not enable_text or caption_key in row) and \
        (not enable_image or image_key in row) and \
            (not enable_metadata or metadata_key in row)


def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    metadata_key="json",
    enable_metadata=False,
    cache_path=None
):
    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)

    filter_fn = partial(
        filter_row, caption_key=caption_key, image_key=image_key, metadata_key=metadata_key,
        enable_text=enable_text, enable_image=enable_image, enable_metadata=enable_metadata
    )
    filtered_dataset = dataset.select(filter_fn)

    transform_fn = partial(
        preprocess_dataset, caption_key=caption_key, image_key=image_key,
        metadata_key=metadata_key, image_transform=image_transform, enable_image=enable_image,
        enable_text=enable_text, enable_metadata=enable_metadata
    )
    transformed_dataset = filtered_dataset.map(transform_fn, handler=wds.handlers.warn_and_continue)
    return transformed_dataset
