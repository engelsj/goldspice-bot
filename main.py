import pandas as pd

# pull in data sets and stack them into singular df
df1 = pd.read_csv("whole-breaking-financial.csv")
df1 = df1.loc[df1['Author'] == 'GoldSpice#6624']
df2 = pd.read_csv("whole-der-analysis.csv")
df2 = df2.loc[df2['Author'] == 'GoldSpice#6624']

# concat dfs together and isolate messages
full_set = pd.concat([df1, df2])
full_set = full_set["Content"]
print(full_set)

