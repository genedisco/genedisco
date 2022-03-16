from collections import defaultdict
from unittest import result
import pandas as pd
import glob
import sys



def get_results(data, features, metric):
    results = defaultdict(list)
    for ack in glob.glob(data + '/' + features + '/*'):
        _ack = ack.replace(data + '/' + features + '/', '')
        results[_ack] = defaultdict(list)
        for seed in glob.glob(ack + '/*'):
            _seed = seed.replace(ack + '/', '')
            print(_ack, _seed)
            d = pd.read_pickle(seed + '/results.pickle')
            for cycle in range(len(d)):
                results[_ack][_seed].append(d[cycle][metric])
    return results

def fun(df):
    df_ = pd.DataFrame(columns=df.columns)
    for col in df.columns: #would be nice to do it without a loop
        df_[col]=pd.Series(df[col][0])
    return df_

def get_acq_dfs(df):
    dfs = []
    for acq in df.index:
        acq_df = df[df.index==acq]
        acq_df = fun(acq_df)
        dfs.append(acq_df)
    return dfs



data = sys.argv[1]
features = sys.argv[2]

data = 'data_sanchez_2021_tau'
features = 'feat_achilles'

metric = 'MeanAbsoluteError'
results = get_results(data, features, metric)
df = pd.DataFrame.from_dict(results,orient='index')

acq_dfs=get_acq_dfs(df)


