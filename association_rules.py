import csv

import pandas as pd
from IPython.display import display
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import OnehotTransactions

dataset = []
with open('supermarket.csv', newline='') as f:
    transactions = csv.reader(f)
    dataset = list(transactions)

    oht = OnehotTransactions()
    oht_ary = oht.fit(dataset).transform(dataset)
    data_frame = pd.DataFrame(oht_ary, columns=oht.columns_)

    frequent_itemsets = apriori(data_frame, min_support=0.3, use_colnames=True)

    # get the association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.84).round(2)

display(rules)
