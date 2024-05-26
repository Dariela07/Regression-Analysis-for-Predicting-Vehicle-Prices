import pandas as pd

# create DataFrame
df = pd.DataFrame({'store': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
                   'sales': [12, 15, 24, 24, 14, 19, 12, 38],
                   'refunds': [4, 8, 7, 7, 10, 5, 4, 11]})

print(df.groupby('store').get_group('A')['sales'])
