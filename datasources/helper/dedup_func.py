import json
import pandas as pd

from tqdm import tqdm

# use the provided exchange_ranking.json
def load_exchange_ranking(exchange_ranking_path:str) -> pd.DataFrame:
    with open(exchange_ranking_path, 'r') as f:
        exch_data = json.load(f)

    exchange_df = pd.DataFrame(exch_data)

    new_column_names = {
        "rank": "meta_exchange_rank",
        "code": "meta_exchange",
        "name": "meta_exchange_display_name"
    }

    exchange_df = exchange_df.rename(columns=new_column_names)
    return exchange_df

def name_and_exchange_priority_based_deduplication(df:pd.DataFrame, exchange_ranking_path:str) -> pd.DataFrame:
    exchange_df = load_exchange_ranking(exchange_ranking_path)
    # merge rankings
    # note, clean exchange names are also available in meta_exchange_display_name after this merge
    df_ranked = pd.merge(df, exchange_df, on='meta_exchange', how='left')
    del df, exchange_df
    # sort by assigned rank
    df_ranked = df_ranked.sort_values(by=['key_year', 'meta_name', 'meta_exchange_rank'])

    # fill data both ways to assure that all information is kept

    rfill_group = lambda group: group.ffill().bfill()
    filled_groups = []
    grouped_data_list = []
    c_ = 0

    # alternative itterative approach to avoid excessive RAM build-up of processing all groups at once
    for _, group_ in tqdm(df_ranked.groupby(['key_year', 'meta_name']), desc="Front/Back Fill"):
        filled_groups.append(rfill_group(group_))
        c_ += 1
        
        # priodically append the data
        if c_ % 20000 == 0:
            grouped_data_list.append(pd.concat(filled_groups))
            filled_groups = []

    # concat remaining groups
    grouped_data_list.append(pd.concat(filled_groups))
    grouped_df = pd.concat(grouped_data_list)

    # save step to avoid waiting while in dev
    # grouped_df.to_csv('data/rfilled_data.csv', index=False)

    # keep only data from top priority (highest rank) exchange
    top_priority_df = grouped_df.groupby(['key_year', 'meta_name']).head(1).reset_index(drop=True)

    # keep the other ticker combinations
    # adjust the aggregation to exclude the first ticker
    alternative_ticker_df = grouped_df.groupby(['key_year', 'meta_name']).apply(lambda x: list(set(list(x['key_ticker'])[1:]))).reset_index(name='meta_other_ticker_list')

    # merge back
    top_priority_df = pd.merge(top_priority_df, alternative_ticker_df, on=['key_year', 'meta_name'], how='left')

    # save final data
    # top_priority_df.to_csv('data/dedup_data.csv', index=False)

    return top_priority_df

