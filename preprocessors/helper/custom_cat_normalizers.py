import abc
import io
import json
from base.common import OxariTransformer
from base.oxari_types import ArrayLike
from typing_extensions import Self
from pathlib import Path
from country_converter import CountryConverter
import pandas as pd 
import linktransformer as lt


MODULE_PATH = Path(__file__).absolute().parent
INDUSTRY_MAPPING_PATHS = [
    MODULE_PATH / "ind-factset-mapping.json",
    MODULE_PATH / "ind-gdc-mapping.json",
    MODULE_PATH / "ind-tw-mapping.json"
]

class OxariCatColumnNormalizer(OxariTransformer):

    def __init__(self, col_name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.col_name = col_name

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class SectorNameCatColumnNormalizer(OxariCatColumnNormalizer):

    def __init__(self, col_name="ft_catm_sector_name", path_to_mapping=MODULE_PATH / "gcis_sector-mapping.json", **kwargs) -> None:
        super().__init__(col_name, **kwargs)
        self.path_mapping = path_to_mapping

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        with io.open(self.path_mapping, "r") as f:
            tmp_mapping:dict[str,dict[str,str]] = json.load(f)
            self.mapping = {key.lower().strip() :v.get("gcis_sector", "").lower().strip() for key, v in tmp_mapping.items()}
        return self
    
    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_new[self.col_name] = X_new[self.col_name].astype(str).str.lower().str.strip().replace(self.mapping) 
        return X_new
    

class LinkTransformerCatColumnNormalizer(OxariCatColumnNormalizer):
    DEFAULT = 'sentence-transformers/all-MiniLM-L6-v2'
    DEFAULT_LG = 'sentence-transformers/all-MiniLM-L12-v2'
    GTE_SMALL = "thenlper/gte-small"
    E5_BASE = "intfloat/e5-base-v2"


    def __init__(self, col_name=["ft_catm_sector_name", "ft_catm_industry_name"], path_to_mapping=MODULE_PATH / "gics_mod.csv", lt_model=DEFAULT, **kwargs) -> None:
        super().__init__(col_name, **kwargs)
        self.path_mapping = path_to_mapping
        self.lt_model = lt_model

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        self.mapping = pd.read_csv(self.path_mapping)
        return self
    
    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_original = X.copy()
        self.mapping["merge_col"] = self.mapping[self.col_name].astype('str').agg(' @ '.join, axis=1)
        X_original["merge_col"] = X_original[self.col_name].astype('str').agg(' @ '.join, axis=1)
        # TODO: There's a smaller and better model here: https://huggingface.co/thenlper/gte-small
        after_merge = lt.merge(X_original, self.mapping, merge_type='1:1', on='merge_col', model=self.lt_model)
        thresholds_reached = after_merge["score"] >= 0.5
        X_new.loc[thresholds_reached.values, self.col_name] = after_merge.loc[thresholds_reached.values, [col+"_y" for col in self.col_name]].values
        return X_new


class IndustryNameCatColumnNormalizer(OxariCatColumnNormalizer):

    def __init__(self, col_name = "ft_catm_industry_name", paths_to_mapping = INDUSTRY_MAPPING_PATHS, **kwargs) -> None:
        super().__init__(col_name, **kwargs)
        self.paths_mapping = paths_to_mapping
        self.mapping: dict[str, str] = {}

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        for path in self.paths_mapping:
            with io.open(path, "r") as f:
                tmp_mapping: dict[str, dict[str, str]] = json.load(f)
                self.mapping.update({key.lower().strip() :v.get("gcis_industry", "").lower().strip() for key, v in tmp_mapping.items()})
        return self
    
    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_new[self.col_name] = X_new[self.col_name].astype(str).str.lower().str.strip().replace(self.mapping) 
        return X_new


class CountryCodeCatColumnNormalizer(OxariCatColumnNormalizer):

    def __init__(self, col_name = "ft_catm_country_code", code_from = "ISO2", code_to = "ISO3", **kwargs) -> None:
        super().__init__(col_name, **kwargs)
        self.mapping: dict[str, str] = {}
        self.code_from = code_from
        self.code_to = code_to

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        correspondence = CountryConverter().get_correspondence_dict(self.code_from, self.code_to)
        self.mapping = {key.lower().strip() :v[0].lower().strip() for key, v in correspondence.items()}
        return self
    
    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_new[self.col_name] = X_new[self.col_name].astype(str).str.lower().str.strip().replace(self.mapping) 
        return X_new
    
    
class OxariCategoricalNormalizer(OxariTransformer, abc.ABC):
    def __init__(self, col_transformers:list[OxariCatColumnNormalizer]=[], **kwargs) -> None:
        super().__init__(**kwargs)
        self.col_transformers = col_transformers

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        for transformer in self.col_transformers:
            transformer.fit(X, y)
        return self
    
    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.copy()
        for transformer in self.col_transformers:
            X_new = transformer.transform(X_new)
        return X_new
