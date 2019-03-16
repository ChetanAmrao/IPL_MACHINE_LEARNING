import pandas as pd
import numpy as np

path = "../ipl1/matches.csv"
team_dict = {}
df = pd.read_csv(path)
print(df.head())

df = np.asarray(df)
nIndex,jIndex = df.shape
k = 0
for i in range(1,nIndex):
	key = df[i,4]
	if key in team_dict:
		print("old key %s"%key)
	else:
		print("NEW KEY %s"%key)
		team_dict[key] = k
		k = k + 1
print(nIndex,jIndex)
print(team_dict)


x = np.zeros((nIndex,4))
x[:,0] = df[:,4]
print(x)