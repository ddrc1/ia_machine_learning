from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

data = pd.read_csv('bag_of_words.csv')
print(data["classificacao"].value_counts())

data = shuffle(data)
x = data.drop(["classificacao"], axis = 1)
y = data["classificacao"]

logreg = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial', max_iter=10000).fit(x[:1500], y[:1500])
# # print(data)
# # print(clf.predict(x))
precision = logreg.score(x[1500:], y[1500:])
print("precision", precision)
conf_mat = confusion_matrix(y[1500:], logreg.predict(x[1500:]))
print("-----------")
print("y = correto : x = predito")
print(conf_mat)
true_positives = conf_mat[0][0]
positives = conf_mat[0][0] + conf_mat[0][1]
revocacao = true_positives / positives
print("revocacao:", revocacao)
f1_score = 2 * ((precision * revocacao) / (precision + revocacao))
print("f1_score:", f1_score)