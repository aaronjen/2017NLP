import pandas as pd
import json

predicts = json.load(open('predicts.json'))

df = pd.read_csv('data/test.csv')

df['Label'] = df.apply(lambda x: predicts[str(x['Review_id'])][x['Aspect']], axis=1)

df.to_csv('sub.csv', columns=['Id', 'Label'], index=False)