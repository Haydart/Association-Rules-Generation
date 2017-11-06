import csv

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import OnehotTransactions

dataset = []
with open('supermarket.csv', newline='') as f:
    transactions = csv.reader(f)
    dataset = list(transactions)

    oht = OnehotTransactions()
    oht_ary = oht.fit(dataset).transform(dataset)
    data_frame = pd.DataFrame(oht_ary, columns=oht.columns_)

    # apriori calculate the frequent items sets in provided data
    frequent_itemsets = apriori(data_frame, min_support=0.3, use_colnames=True)

    # round supports to 2 decimal places
    frequent_itemsets['support'] = frequent_itemsets['support'].apply(lambda x: round(x, 2))

    # create additional column for sorting
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # sort the values

    print(frequent_itemsets)
