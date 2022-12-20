"""Templates to create class embedding"""
import pickle
from typing import Any, Dict, List, Union

import fsspec
import pyspark
import torch
from clip_on_yarn.config import CONFIG
from clip_on_yarn.utils.translate import load_m2m100_12B
from clip_on_yarn.utils.uc import CAT_LANGUAGES_OF_INTEREST, Language, Taxonomy, filter_taxonomy_to_keep_last_level
from easynmt import EasyNMT
from ml_hadoop_experiment.common.spark_inference import SerializableObj
from ml_hadoop_experiment.pytorch.spark_inference import with_inference_column_and_preprocessing
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType
from thx.hadoop import hdfs_cache
from thx.hadoop.spark_config_builder import create_remote_spark_session

TEMPLATES_HDFS_PATH = "viewfs://root/user/r.fabre/multi_lang_captions_per_category"
TEMPLATES_EN = [
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
]

TAXONOMY = filter_taxonomy_to_keep_last_level(Taxonomy.build(language=Language.en_US))


def _create_and_translate_captions(
    translation_model: EasyNMT, features: Union[List[torch.Tensor], torch.Tensor], device: str
) -> List[Any]:
    """Generate captions and translate them in target languages"""
    translation_model.device = device
    google_categories_lvl_4 = features[0].numpy()
    all_captions = []
    for google_category_lvl_4 in google_categories_lvl_4:
        print(f"Processing google_category_lvl_4: {google_category_lvl_4}")
        category_name = TAXONOMY.category_id_to_category[google_category_lvl_4].name
        captions = {"en": [f(category_name) for f in TEMPLATES_EN]}
        with torch.inference_mode():
            for lang in CAT_LANGUAGES_OF_INTEREST:
                if lang == "en":
                    continue
                captions[lang] = translation_model.translate(
                    captions["en"], source_lang="en", target_lang=lang, batch_size=4
                )
        all_captions.append([captions[lang] for lang in CAT_LANGUAGES_OF_INTEREST])
    return all_captions


def with_multi_lang_caption_column(
    ss, df: pyspark.sql.dataframe.DataFrame, google_cat_level_4_col_name: str
) -> pyspark.sql.dataframe.DataFrame:
    serializable_translation_model = SerializableObj(ss, load_m2m100_12B, ".")

    return with_inference_column_and_preprocessing(
        df=df,
        artifacts=serializable_translation_model,  # type: ignore
        input_cols=[google_cat_level_4_col_name],
        preprocessing=lambda _, x, __: x,  # type: ignore
        inference_fn=_create_and_translate_captions,  # type: ignore
        output_type=ArrayType(ArrayType(StringType())),
        batch_size=200,
        output_col="multi_lang_captions",
        num_threads=4,
        num_workers_preprocessing=4,
    )


def create_templates_for_uc_lang() -> None:
    """Create CLIP templates for all languages in our catalogue"""
    if not hdfs_cache.exists("/user/r.fabre/multi_lang_captions_per_category"):
        num_cores = 80
        memory = 32
        num_gpu_machines = 30
        num_gpus_per_machine = 2
        num_containers = num_gpu_machines * num_gpus_per_machine
        num_cores_per_tasks = int(num_cores / num_gpus_per_machine)
        properties = [
            ("spark.task.cpus", str(num_cores_per_tasks)),
            ("spark.speculation", "true"),
            ("spark.speculation.interval", "10s"),
            ("spark.speculation.multiplier", "13"),
            ("spark.speculation.quantile", "0.10"),
            ("spark.task.maxFailures", "100"),
            ("spark.blacklist.enabled", "false"),
            ("spark.yarn.max.executor.failures", str(1000)),
            ("spark.yarn.queue", "ml-gpu"),
            ("spark.yarn.executor.nodeLabelExpression", "gpu"),
        ]

        memory_heap = int(memory * 0.25)
        memory_offheap = int(memory * 0.75)

        spark_params: Dict[str, Any] = {
            "app_name": "Create CLIP templates",
            "num_containers": num_containers,
            "num_cores": num_cores,
            "memory": str(memory_heap) + "g",
            "memoryOverhead": str(memory_offheap) + "g",
            "properties": properties,
            "hadoop_file_systems": "viewfs://root,viewfs://prod-am6,viewfs://preprod-am6,hdfs://root",
        }
        ss = create_remote_spark_session(**spark_params)
        uc_lv4_categories = [[cat.id] for cat in TAXONOMY.categories]
        schema = StructType([StructField("uc_id", IntegerType(), True)])
        df_uc_lv4 = ss.createDataFrame(uc_lv4_categories, schema=schema)
        captions_df = with_multi_lang_caption_column(ss, df_uc_lv4, "uc_id")
        captions_df.write.mode("overwrite").parquet(TEMPLATES_HDFS_PATH)
        ss.stop()


def create_templates_per_lang_x_uc_id() -> None:
    """Generate and save CLIP prompt templates for all languages of interest"""
    if not hdfs_cache.exists(CONFIG.templates_per_lang_x_uc_id_path):
        create_templates_for_uc_lang()
        spark_params: Dict[str, Any] = {
            "app_name": "Generate CLIP templates",
            "num_containers": 10,
            "num_cores": 4,
        }
        ss = create_remote_spark_session(**spark_params)
        df_templates = ss.read.parquet(TEMPLATES_HDFS_PATH).toPandas()
        ss.stop()
        templates_per_lang_x_uc_id: Dict[str, Dict[int, List[str]]] = {}
        for i, lang in enumerate(CAT_LANGUAGES_OF_INTEREST):
            templates_per_lang_x_uc_id[lang] = {}
            for uc_id, templates in df_templates.values:
                templates_per_lang_x_uc_id[lang][uc_id] = templates[i]
        fs = fsspec.filesystem("hdfs")
        with fs.open(CONFIG.templates_per_lang_x_uc_id_path, "wb") as f:
            pickle.dump(templates_per_lang_x_uc_id, f, protocol=pickle.HIGHEST_PROTOCOL)


def create_uc_id_to_idx_mapping() -> None:
    """Encode target labels with value between 0 and n_classes-1 and save the mapping"""
    fs = fsspec.filesystem("hdfs")
    if not hdfs_cache.exists(CONFIG.uc_id_to_idx_mapping_path):
        create_templates_for_uc_lang()
        spark_params: Dict[str, Any] = {
            "app_name": "Generate class indexing",
            "num_containers": 10,
            "num_cores": 4,
        }
        ss = create_remote_spark_session(**spark_params)
        df_templates = ss.read.parquet(TEMPLATES_HDFS_PATH).select("uc_id").toPandas()
        ss.stop()
        uc_id_to_idx_mapping = {uc_id: idx for idx, uc_id in enumerate(df_templates.uc_id)}
        with fs.open(CONFIG.uc_id_to_idx_mapping_path, "wb") as f:
            pickle.dump(uc_id_to_idx_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
