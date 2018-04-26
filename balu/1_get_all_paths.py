from glob import glob
import pandas as pd

folders = glob('../data/train/*')

X = []
y = []

for folder in folders:
	images = glob(folder+'/*.jpg')
	X += images

	name = folder.split('/')[-1]
	print(name)
	y += [name for x in images]


print len(X)
print len(y)

df = pd.DataFrame()
df['path'] = pd.Series(X)
df['label'] = pd.Series(y)
df.to_csv('../data/csv/all.csv',index=False)
