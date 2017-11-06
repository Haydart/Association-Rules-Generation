# coding: utf-8

# In[1]:


# generating frequent_itemsets as in previous example
import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
# generating frequent_itemsets as in previous example
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import OnehotTransactions

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

oht = OnehotTransactions()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

frequent_itemsets

# In[63]:


from mlxtend.frequent_patterns import association_rules

# show all rules created by apriori with threshold set to 60%
association_rules(frequent_itemsets, metric='support', min_threshold=0.0)  # here min_threshold=0.0 because aprori has
# used a threshold


# In[82]:


# mine rules with confidence >= 75%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.74)  # be aware of rounding!
# 0.74999999 vs 0.75
rules.sort_values(by='confidence', ascending=False)  # sort results

# In[83]:


# interested in rules having lift > 1.00, confidence > 0.75 and 2 antecedants
rules['antecendant_len'] = rules['antecedants'].apply(lambda x: len(x))
rules

rules[(rules['lift'] > 1.0) &
      (rules['confidence'] > 0.75) &
      (rules['antecendant_len'] == 2)]
