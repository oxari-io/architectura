import abc
import io
import json
from base.common import OxariTransformer
from base.oxari_types import ArrayLike
from typing_extensions import Self
from pathlib import Path

MODULE_PATH = Path(__file__).absolute().parent


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
