
import cpuinfo
from dotenv import load_dotenv
import pandas as pd 

load_dotenv()

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('max_colwidth',20)
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 10)

from .common import *
from .confidence_intervall_estimator import *
from .dataset_loader import *
