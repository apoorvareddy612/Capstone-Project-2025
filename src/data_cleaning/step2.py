import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import contractions
import re
'''Text cleaning and formatting'''

#Huberman Lab
df = pd.read_csv('./data/huberman.csv')
df1 = pd.read_csv('./data/lex_fridman.csv')
df2 = pd.read_csv('./data/skeptoid.csv')
df3 = pd.read_csv('./data/vox.csv')
#add new column which addes the name of csv files
df['source'] = 'huberman'
df1['source'] = 'lex_fridman'
df2['source'] = 'skeptoid'
df3['source'] = 'vox'
#rename the columns
df3.rename(columns={'episodeName':'title'},inplace=True)
#concatenate all the dataframes
df = pd.concat([df,df1,df2,df3],ignore_index=True)
print(df.head())
df.to_csv('./data/data.csv',index=False)
