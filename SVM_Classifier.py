import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.utils import shuffle

features = list()
df = pd.read_csv('class_data.csv')

for index in range(df.shape[0]):
	text_file = ' '.join([word for word in df.ix[index,'lyrics'].split()])
	features.append(text_file)
labels = df['label'].tolist()
print('training data loaded!')

features2 = list()
df2 = pd.read_csv('lyrics_generated.csv')

for index in range(df2.shape[0]):
	text_file2 = ' '.join([word for word in df2.ix[index,'lyrics'].split()])
	features2.append(text_file2)
labels2 = df2['label'].tolist()
print('test data loaded!')

idx = shuffle(range(len(features)))
features = [features[i] for i in idx]
labels = [labels[i] for i in idx]

print('Start training...')
classifier = SVC(kernel = 'linear', probability = True)
text_clf = Pipeline([('vect', CountVectorizer()),
					('tfidf', TfidfTransformer(use_idf = True)),
					# ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
					('clf', classifier)])
text_clf.fit(features, labels)
print('Training finished!')

print('Start testing...')
test_pred = text_clf.predict(features2)
results = text_clf.predict_proba(features2)
for result in results:
	prob_per_class_dictionary = dict(zip(text_clf.classes_, result))
	print(prob_per_class_dictionary)
with open('result.csv', 'w') as result:
	result.write("Lable,Predict\n")
	for i in range(len(labels2)):
		result.write(str(labels2[i]) + ',' + str(test_pred[i]) + '\n')
print('Testing finished!')

# final = []
# kf = KFold(n_splits = 5)
# print('-'*5 + '5 Folds Result' + '-'*5)
# for train_idx, test_idx in kf.split(features):
# 	train_features = [features[i] for i in train_idx]
# 	train_labels = [labels[i] for i in train_idx]
# 	test_features = [features[i] for i in test_idx]
# 	test_labels = [labels[i] for i in test_idx]
# 	text_clf.fit(train_features, train_labels)
# 	tmp = np.mean(text_clf.predict(test_features) == test_labels)
# 	print(tmp)
# 	final.append(tmp)

# print('-'*5 + 'Average Result' + '-'*5)
# print(np.mean(np.array(final)))



