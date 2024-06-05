

import pandas as pd
from base import OxariDataManager, OxariImputer
from base.run_utils import get_small_datamanager_configuration
from imputers import BaselineImputer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm

from imputers.core import DummyImputer
from imputers.equilibrium_method import EquilibriumImputer, FastEquilibriumImputer


if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = get_small_datamanager_configuration(0.35).run()
    configurations: list[OxariImputer] = [
        BaselineImputer(),
        *[EquilibriumImputer(verbose=False, mims_tresh=m, skip_converged_cols=b, diff_tresh=0, max_diff_increase_thresh=0.6) for m in [0.01, 0.001, 0.00001] for b in [True, False]],
        *[FastEquilibriumImputer(verbose=False, mims_tresh=m, skip_converged_cols=b, diff_tresh=0, max_diff_increase_thresh=0.6) for m in [0.01, 0.001, 0.00001] for b in [True, False]]
    ]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations) * len(difficulties)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_new = X.copy()
            X_new[X.filter(regex='^ft_num', axis=1).columns] = minmax_scale(X.filter(regex='^ft_num', axis=1))

            X_train, X_test = train_test_split(X_new, test_size=0.5)
            keeping_criterion_2 = (X_test.isna().mean(axis=0)<0.3)
            keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

            for imputer in configurations:
                imputer_2: OxariImputer = imputer.clone()
                imputer_2 = imputer_2.fit(X_train[keep_columns_2])

                for dff in difficulties:

                    imputer_2.evaluate(X_test[keep_columns_2], p=dff)
                    all_results.append({"repetition": i, "difficulty": dff, "mode":"low_missingness", "num_ft":len(keep_columns_2),**imputer_2.evaluation_results, **imputer_2.get_config()})

                    concatenated = pd.json_normalize(all_results)
                    fname = __loader__.name.split(".")[-1]
                    pbar.update(1)
                    concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')







