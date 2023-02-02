from pathlib import Path

import pandas as pd
from typing_extensions import Self

from base.dataset_loader import Datasource


class LocalDatasource(Datasource):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(self.path)

    def _check_if_data_exists(self):
        if not self.path.exists():
            self.logger.error(f"Exception: Path(s) does not exist! Got {self.path}")
            raise Exception(f"Path(s) does not exist! Got {self.path}")

    def _load(self) -> Self:
        self._data = pd.read_csv(self.path)
        return self



