from empath import Empath
import arff
import pandas as pd

data = arff.load(open('IMDB.arff', 'r'))
print(data.keys())
print(data['attributes'])

emp = Empath()
results = []
classifs = []
for i, line in enumerate(data["data"]):
    comment = line[0]
    classif = line[1]
    results.append(emp.analyze(doc=comment, normalize=False))
    classifs.append(classif)
print(len(results))
df = pd.DataFrame(results)
df["classificacao"] = classifs
df.to_csv("imdb_empath.csv", index=False)

