import pandas as pd
import numpy as np

path = "../ipl1/modifiedMATCHES.csv"
team_dict = {}
df = pd.read_csv(path)

df = np.asarray(df)
nIndex,jIndex = df.shape
k = 0
x = np.zeros((nIndex,4))
y = np.zeros((nIndex,1))
for i in range(1,nIndex):
	key = df[i,4]
	if key in team_dict:
		continue
	else:
		team_dict[key] = k
		k = k + 1
for j in range(0,3):
	for i in range(0,nIndex):
		key = df[i,j+4]
		x[i,j] = team_dict[key]
		print(key,team_dict[key])
	print("---------------------------------------------------------------\n\n\n")

for i in range(0, nIndex):
	key = df[i,10]
	if (key == "tie"):
		y[i,0] = -1
	else:
		y[i,0] = team_dict[key]

print(y)
