import pandas as pd
import ast

# navigate to  C:\Users\condo\OneDrive\Documents\PhD before running

### adhoc, can delete after.
from sklearn.model_selection import train_test_split
train = pd.read_csv('C:/Users/condo/Downloads/LinnData/LinnData/MIFreq_KI_train.csv')
val = pd.read_csv('C:/Users/condo/Downloads/LinnData/LinnData/MIFreq_KI_val.csv')
test = pd.read_csv('C:/Users/condo/Downloads/LinnData/LinnData/MIFreq_KI_test.csv')

test2 = pd.concat([val, test])


# correct PHRASES - top_temp.csv
df = pd.read_csv('/Users/condo/20230112-122104.csv')
df = df.tail(150000)


# both phrases
df = pd.read_csv('/Users/condo/20230115-061121.csv')
df = df.tail(75000)
#df_top = df[df['scores'] > 1.1]

# bad phrases
df = pd.read_csv('/Users/condo/20230113-091908.csv')
df = df.tail(150000)


# actions less than 30: adding a phrase
df = df.sort_values(by='scores', ascending=False)
df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x))
df['action_len'] = df['actions'].apply(lambda x: len(x))



# len(df[df['scores'] > -1]) / len(df)
df_top = df[df['scores'] > 2.24]  # for good phrases, 3.1. -1.01 for bad phrases, -1, -1.005 for both
df_top['actions'] = df_top['actions'].apply(lambda x: str(x))
df_top = df_top.fillna(0)
top_unique = df_top.groupby(['0', '1', '2', '3', '4', '5', '6', 'scores', 'actions', 'action_len']).mean().reset_index()

# filter for any repeated actions
top_unique['actions'] = top_unique['actions'].apply(lambda x: ast.literal_eval(x))
top_unique['set_len'] = top_unique['actions'].apply(lambda x: len(set(x)))
top_unique['duplicate'] = top_unique.apply(lambda x: x['action_len'] == x['set_len'], axis=1)
top_dups = top_unique[top_unique['duplicate'] == False]
top_dups.to_csv('top_dups.csv')

# how many added needed only one action
top_unique['action_len'].value_counts()

# when only one action chosen which is it
top_unique[top_unique['action_len'] == 1]['actions'].value_counts().head(15)
# when two actions chosen which combinations top 15
top_unique[top_unique['action_len'] == 2]['actions'].value_counts().head(15)
# when two actions chosen which combinations top 15
top_unique[top_unique['action_len'] == 3]['actions'].value_counts().head(10)

##### data set with individual words
df = pd.read_csv('/Users/condo/20221228-122636.csv')
df = df.sort_values(by='scores', ascending=False)
df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x))
df['action_len'] = df['actions'].apply(lambda x: len(x))

# top12percent
top_scores = df[df['scores'] > -.95]
top_scores['action_len'].value_counts()
top_scores[top_scores['action_len'] == 1]['actions'].value_counts()
top_scores[top_scores['action_len'] == 2]['actions'].value_counts().head(15)
top_scores[top_scores['action_len'] == 3]['actions'].value_counts().head(15)
top_scores[top_scores['action_len'] == 4]['actions'].value_counts().head(15)

