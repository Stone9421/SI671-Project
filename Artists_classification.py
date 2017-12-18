import csv
import pandas as pd
import random

df = pd.read_csv('lyrics.csv')

df = df.drop(['index', 'song', 'year'], axis = 1)

df = df.loc[(df['genre'] == 'Pop') | 
			(df['genre'] == 'Rock') | 
			(df['genre'] == 'R&B') | 
			(df['genre'] == 'Country') | 
			(df['genre'] == 'Hip-Hop') | 
			(df['genre'] == 'Metal')]

df = df.dropna()

df_1 = df.groupby(['artist', 'genre'])\
		 .size()\
		 .reset_index(name = 'counts')\
		 .sort_values(by = ['genre', 'counts'], ascending = [True, False])\
		 .groupby('genre')\
		 .head(5)

# print(df_1)

df_2 = df.loc[df['artist'].isin(df_1['artist'])]
ll = pd.DataFrame(columns=['artist','genre','lyrics','label'])
cc = pd.DataFrame(columns=['artist','genre','lyrics','label'])
c = 1
for artist in df_2['artist'].unique():
	each_artist_df = df_2[df_2['artist'] == artist] # dataframe for each artist 
	each_artist_song_number = len(each_artist_df.index) # total number of songs
	try:
		lstm_tr = random.sample(each_artist_df.index.tolist(), round(each_artist_song_number/2))
		lstm = each_artist_df.loc[lstm_tr]
		clas = each_artist_df.loc[~each_artist_df.index.isin(lstm_tr)]
		lstm['label'] = c
		clas['label'] = c
		ll = ll.append(lstm)
		cc = cc.append(clas)
		c += 1
	except:
		pass
cc.to_csv('class_data.csv', index = False)
ll.to_csv('lstm_data.csv', index = False)
