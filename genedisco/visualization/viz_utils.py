from collections import defaultdict
from unittest import result
import pandas as pd
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt



def get_results(data, features, metric):
    results = defaultdict(list)
    for ack in glob.glob(data + '/' + features + '/*'):
        _ack = ack.replace(data + '/' + features + '/', '')
        #_ack =_ack=_ack.split('\\')[-1] #keep the line for windows, remove for linux
        results[_ack] = defaultdict(list)
        for seed in glob.glob(ack + '/*'):
            _seed = seed.replace(ack + '/', '')
            #_seed = _seed.split('\\')[-1] #keep the line for windows, remove for linux
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

def plot_acq(df,acq_name,color):
    fig, subplot = plt.subplots()
    D=df.to_numpy()
    sems = np.std(D, axis=1, ddof=1) / np.sqrt(np.size(D))
    means =np.mean(D, axis = 1)

    
    plt.xlabel('cycles')
    plt.ylabel('metric')
    plt.title(f'Mean and stderr over cycles for {acq_name} acquisition function')

    subplot.plot([0,1,2,3,4,5,6,7],
                        means,
                        lw=6,
                        ls=":", label=acq_name, color=color
                    )

    subplot.fill_between(x=[0,1,2,3,4,5,6,7],
                        y1=means - 1.96 * sems,
                        y2=means + 1.96 * sems,
                        alpha=0.3, color=color
                    )
    subplot.legend(loc='best')
    fig.savefig(f'plot_{acq_name}.jpg')


data = sys.argv[1]
features = sys.argv[2]
metric = sys.argv[3]

#data = 'data_sanchez_2021_tau'
#features = 'feat_achilles'
#metric = 'MeanAbsoluteError'

results = get_results(data, features, metric)
df = pd.DataFrame.from_dict(results, orient='index')

acq_dfs=get_acq_dfs(df)

colors=['red','greenyellow','violet','cyan','yellow','blue','magenta','lime','gold']
for i in range(len(acq_dfs)):
    plot_acq(acq_dfs[i],df.index[i],colors[i])
