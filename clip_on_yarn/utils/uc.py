"""Utils related to the universal catalogue"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import requests

CAT_LANGUAGES_OF_INTEREST = ["en", "de", "fr", "it", "ja", "nl", "es"]


class Language(str, Enum):
    """Catalogue languages"""

    cs_CZ = "cs-CZ"
    da_DK = "da-DK"
    de_DE = "de-DE"
    en_US = "en-US"
    es_ES = "es-ES"
    fr_FR = "fr-FR"
    id_ID = "id-ID"
    it_IT = "it-IT"
    ja_JP = "ja-JP"
    kr_KR = "kr-KR"
    nl_NL = "nl-NL"
    no_NO = "no-NO"
    pl_PL = "pl-PL"
    pt_PT = "pt-PT"
    ru_RU = "ru-RU"
    sv_SE = "sv-SE"
    tr_TR = "tr-TR"
    vi_VN = "vi-VN"
    zh_CN = "zh-CN"


@dataclass
class Category:
    id: int
    name: str
    ancestors: List[int]

    @property
    def level(self) -> int:
        return len(self.ancestors) + 1


def _extract_category(line: str, categories_by_name: Dict[str, Category]) -> Category:
    id_str, categories_str = line.split(" - ", 1)
    categories = categories_str.split(" > ")
    ancestors = [categories_by_name[ancestor_name].id for ancestor_name in categories[:-1]]
    return Category(int(id_str), categories[-1], ancestors)


def filter_taxonomy_to_keep_last_level(taxonomy: "Taxonomy", max_level: int = 4) -> "Taxonomy":
    candidates = [c for c in taxonomy.categories if c.level <= max_level]
    ancestors = set()
    for c in candidates:
        ancestors.update(c.ancestors)
    return Taxonomy([c for c in candidates if c.id not in ancestors])


class Taxonomy:
    """UC taxonomy tree"""

    def __init__(self, categories: List[Category]) -> None:
        self.categories = categories
        self.category_id_to_category: Dict[int, Category] = {category.id: category for category in categories}
        ancestors = {ancestor for c in categories for ancestor in c.ancestors}
        self.leaves = {c.id for c in self.categories if c.id not in ancestors}

    @classmethod
    def build(cls, language: Language = Language.en_US) -> "Taxonomy":
        """Fetch and create taxonomy"""
        taxonomy_url = (
            "https://review.crto.in/gitweb?p=catalog/catalog-api.git;a=blob_plain;f=catalog-api"
            f"/src/main/resources/taxonomy/taxonomy-with-ids.{language.value}.txt"
        )
        raw = requests.get(taxonomy_url, timeout=10).content.decode().strip()
        categories_by_name: Dict[str, Category] = {}
        for line in raw.split("\n")[1:]:
            category = _extract_category(line, categories_by_name)
            categories_by_name[category.name] = category
        return cls(list(categories_by_name.values()))
