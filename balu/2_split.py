from sklearn.model_selection import StratifiedKFold
from glob import glob
import pandas as pd

df = pd.read_csv('../data/csv/all.csv')
X = df['path']
y = df['label']

skf = StratifiedKFold(n_splits=4,random_state=42)

for i,(train, test) in enumerate(skf.split(X, y)):
	
	print (train.shape)
	print (test.shape)

	trainX = pd.Series(X)[train]
	trainY = pd.Series(y)[train]

	df = pd.DataFrame()
	df['path'] = trainX
	df['label'] = trainY
	df.to_csv('../data/csv/train_fold_'+str(i)+'.csv',index=False)

	testX = pd.Series(X)[test]
	testY = pd.Series(y)[test]

	df = pd.DataFrame()
	df['path'] = testX
	df['label'] = testY
	df.to_csv('../data/csv/test_fold_'+str(i)+'.csv',index=False)