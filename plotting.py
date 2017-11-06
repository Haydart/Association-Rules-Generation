import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib')
matplotlib.rcParams['figure.figsize'] = [20, 9]


def transactions_to_heat_map(data_frame):
    heat_map_data = pd.DataFrame(data=np.zeros((len(data_frame.columns), len(data_frame.columns))),
                                 index=data_frame.columns, columns=data_frame.columns)
    for row in data_frame.iterrows():
        purchase = row[1][row[1] > 0]
        for elem1 in purchase.iteritems():
            for elem2 in purchase.iteritems():
                heat_map_data.loc[elem1[0]][elem2[0]] += 1

    sns.heatmap(heat_map_data, annot=True)


def transactions_to_histogram(data_frame):
    histogram_data = pd.DataFrame(data=np.zeros(len(data_frame.columns)), index=data_frame.columns,
                                  columns=['count'], dtype=np.int32)
    for row in data_frame.iterrows():
        purchase = row[1][row[1] > 0]
        for elem in purchase.iteritems():
            histogram_data.loc[elem[0]] += 1

    histogram_data.plot(kind='bar', legend=False)
