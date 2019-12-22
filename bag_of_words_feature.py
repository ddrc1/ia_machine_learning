from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import arff

df = pd.DataFrame(arff.load(open('IMDB.arff', 'r'))["data"])
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df[0])

df2 = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
df2["classificacao"] = df[1]
df2.to_csv('bag_of_words.csv', index = False)