import pandas as pd
from base.dataset_loader import OxariDataManager

from base.run_utils import get_default_datamanager_configuration

dataset: OxariDataManager = get_default_datamanager_configuration().run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
CMP = pd.read_csv('model-data/output/T20230818_p_companies.csv', index_col=0).reset_index()
SCP = pd.read_csv('model-data/output/T20230818_p_scope_imputations.csv', index_col=0).reset_index()
LAR = pd.read_csv('model-data/output/T20230818_p_lar_imputations.csv', index_col=0).reset_index()
DATA_DF = DATA.drop(['index', 'tg_numc_scope_1', 'tg_numc_scope_2', 'tg_numc_scope_3'], axis=1).merge(SCP.drop('index', axis=1),
                                                                                                              on=['key_isin', 'key_year'],
                                                                                                              suffixes=('', '_DROP'))
# DATA_DF = DATA_DF.merge(LAR.drop('index', axis=1), how='left', on=['key_isin', 'key_year'], suffixes=('', '_DROP'))
DATA_DF = DATA_DF.merge(CMP.drop('index', axis=1), how='left', on=['key_isin'], suffixes=('', '_DROP'))
DATA_CLEANED = DATA_DF.drop(DATA_DF.filter(regex='_DROP').columns, axis=1)
DATA_CLEANED_FINAL = DATA_CLEANED.drop(DATA_CLEANED.filter(regex="\.1").columns, axis=1)
DATA_CLEANED_FINAL.to_csv('all_combined.csv')