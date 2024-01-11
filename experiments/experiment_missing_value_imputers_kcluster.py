# pip install autoimpute

import pandas as pd
from base import OxariDataManager, OxariImputer
from base.run_utils import get_small_datamanager_configuration
from imputers import BaselineImputer
from sklearn.model_selection import train_test_split
import tqdm

from imputers.kcluster_bucket import KMeansBucketImputer, KNNBucketImputer, KMedianBucketImputer


if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = get_small_datamanager_configuration(0.5).run()
    configurations: list[OxariImputer] = [
        BaselineImputer(),
        *[KMeansBucketImputer(num_buckets=num) for num in [3, 5, 7, 9, 11, 13, 15]],
        *[KMedianBucketImputer(num_buckets=num) for num in [3, 5, 7, 9, 11, 13, 15]],
        *[KNNBucketImputer(num_buckets=num) for num in [3, 5, 7, 9, 11, 13, 15]],
    ]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations) * len(difficulties)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_new: pd.DataFrame = X.copy()

            X_train, X_test = train_test_split(X_new, test_size=0.5)
            keeping_criterion_2 = (X_test.isna().mean(axis=0) < 0.3)
            keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

            for imputer in configurations:
                imputer.fit(X_train[keep_columns_2])

                for dff in difficulties:

                    imputer.evaluate(X_test[keep_columns_2], p=dff)
                    all_results.append({
                        "repetition": i,
                        "difficulty": dff,
                        "num_ft": len(keep_columns_2),
                        **imputer.evaluation_results,
                        **imputer.get_config()
                    })

                    concatenated = pd.json_normalize(all_results)
                    fname = __loader__.name.split(".")[-1]
                    pbar.update(1)
                    concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')
