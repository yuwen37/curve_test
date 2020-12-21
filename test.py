import pandas as pd
import numpy as np
"""
df.drop_duplicates(subset=['A', 'B'], keep='first', inplace=True)
代码中subset对应的值是列名，表示只考虑这两列，将这两列对应值相同的行进行去重。
默认值为subset=None表示考虑所有列。

keep='first'表示保留第一次出现的重复行，是默认值。keep另外两个取值为"last"和False，
分别表示保留最后一次出现的重复行和去除所有重复行。

inplace=True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本。
"""

"""
DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
"""
x = [1, 2, 3, 4, 5]
y = [1, 1, 2, 2, 3]
df = pd.DataFrame({'x': x, 'y': y})
platform = np.array(df['y'].iloc[1:]) - np.array(df['y'].iloc[:-1])
index = np.where(platform == 0)[0]
df['z'] = df['x'].rolling(2, axis=0).mean()
df['x'].iloc[index+1] = np.array(df['z'].iloc[index+1])
df['x'].iloc[index] = np.array(df['z'].iloc[index+1])
df.drop(['z'], axis=1, inplace=True)
df.drop_duplicates(subset=None, keep='first', inplace=True)
print(df)