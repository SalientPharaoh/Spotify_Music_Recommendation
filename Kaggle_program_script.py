#importing requiered libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity #to compute similarity between two vectors
from sklearn.feature_extraction.text import CountVectorizer #convert text to sparse matrix tokens
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

#loading the dataset required
df=pd.read_csv("/kaggle/input/-spotify-tracks-dataset/dataset.csv")
df.head()

print(df.shape) #understanding the size of dataset
print(df.info()) #column details of dataset

#checking for null values
df.isnull().sum()

#since only one null value, we can remove it as it is very very small proportion of dataset
df.dropna(inplace=True)
df.isnull().sum() #rechecking to confirm

#unnecessary columns are removed
df.drop(['track_id'],axis=1,inplace=True)

#checking for duplicacy in data
print(df.shape)
df['track_name'].nunique()
#result shows that there are around 40000 duplicates in the data

#removing duplicate data from the dataframe
df=df.sort_values(by=['popularity'],ascending=False) #sorting data
df.drop_duplicates(subset=['track_name'],keep='first',inplace=True) #keeping only that copy which is most popular and deleting the rest

#Looking at the dataframe and checking its dimension after removing the duplicates
print(df.shape)
df.head(5)

#converting song names to lower case to avoid problems while taking user inputs
#df['track_name'].apply(lambda s : s.lower())
#df

#Getting all column names
df.columns

#Performing Exploratory Data Analysis
#popularity boxplot of music genres
plt.figure(figsize = (18, 7))
sb.boxplot(data=df.head(50),x='track_genre',y='popularity')
plt.show()

#distribution plots for float data types
flt = []
for i in df.columns:
  if df[i].dtype == 'float':
    flt.append(i)
    
plt.subplots(figsize = (20, 20))
for i, j in enumerate(flt):
  plt.subplot(3, 3, i + 1)
  sb.distplot(df[j])
plt.tight_layout()
plt.show()

vec=CountVectorizer()
vec.fit(df['track_genre'])

df_temp=df.sort_values(by=['popularity'],ascending=False).head(3000)
#converting song name to lowercase to reduce input errors while taking user input
df_temp['track_name']=df_temp['track_name'].str.lower()
df_temp

def similar(song,data):
    txt=vec.transform(data[data['track_name']==song]['track_genre']).toarray()
    num=data[data['track_name']==song].select_dtypes(include=np.number).to_numpy()
    sim = []
    for idx, row in data.iterrows(): 
        txt2 = vec.transform(data[data['track_name']==row['track_name']]['track_genre']).toarray()
        num2 = data[data['track_name']==row['track_name']].select_dtypes(include=np.number).to_numpy()

        
        txtsim = cosine_similarity(txt, txt2)[0][0]
        numsim = cosine_similarity(num, num2)[0][0]
        sim.append(txtsim + numsim)
     
    return sim

def recommend_songs(song, data=df_temp):
    if df_temp[df_temp['track_name'] == song].shape[0] == 0:
        print('Not available in out database!\n Some songs you may like:\n')
        
        for song in data.sample(n=5)['track_name'].values:
          print(song)
        return

    data['similarity_factor'] = similar(song, data)

    data.sort_values(by=['similarity_factor', 'popularity'],ascending = [False, False],inplace=True)

    # First song will be the input song itself as the similarity will be highest.
    return data[['track_name','artists']][2:7]

inp=input("Enter the song name:-\n").lower()
recommendations = recommend_songs(inp).applymap(lambda s : s.capitalize())
recommendations
