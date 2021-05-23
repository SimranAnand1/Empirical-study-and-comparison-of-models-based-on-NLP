#!/usr/bin/env python
# coding: utf-8

# #Simran Anand

# ## <b> Problem Description </b>
# 
# ### This project aims to build a classification model to predict the sentiment of COVID-19 tweets.The tweets have been pulled from Twitter and manual tagging has been done then. Leveraging Natural Language Processing, sentiment analysis is to be done on the dataset. Additionally, machine learning algorithms are to be incorporated to evaluate accuracy score and classification prediction by the trained model.
# 
# ### The following information is used:
# 1. Location
# 2. Tweet At
# 3. Original Tweet
# 4. Label

# ##Importing necessary libraries to build model

# In[63]:


import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import tweepy
from textblob import TextBlob
import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200) 
import string
import branca.colormap as cm
import requests
import folium
from folium import plugins
from folium.plugins import HeatMap
import branca.colormap
import nltk # for text manipulation
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
from wordcloud import WordCloud
from tqdm import tqdm, notebook
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from gensim.models.doc2vec import LabeledSentence
import gensim
from sklearn.linear_model import LogisticRegression
from scipy import stats 
from sklearn import metrics 
from sklearn.metrics import mean_squared_error,mean_absolute_error, make_scorer,classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# ##Extracting dataset and Reviewing Our Dataset

# In[4]:


df=pd.read_csv("https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv")
df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


# There are 12220 unique locations from where the tweets came.
df['user_location'].value_counts()


# # Looking For Null Values

# In[9]:


missing_values = pd.DataFrame()
missing_values['column'] = df.columns

missing_values['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]
missing_values = missing_values.sort_values('percent')
missing_values = missing_values[missing_values['percent']>0]
plt.figure(figsize=(15, 5))
sns.set(style='whitegrid', color_codes=True)
splot=sns.barplot(x='column', y='percent', data=missing_values)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                   va = 'center', xytext = (0, 9), textcoords = 'offset points')
plt.xlabel("Column_Name", size=14, weight="bold")
plt.ylabel("Percentage", size=14, weight="bold")
plt.title("Percentage of missing values in column",fontweight="bold",size=17)
plt.show()


# ##Heat Map for missing values

# In[10]:


plt.figure(figsize=(17, 5))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.xlabel("Column_Name", size=14, weight="bold")
plt.title("Places of missing values in column",fontweight="bold",size=17)
plt.show()


# In[11]:


df.describe()


# In[12]:


sns.heatmap(df.corr())


# ##Unique Values In Each Feature Coulmn

# In[13]:


unique_df = pd.DataFrame()
unique_df['Features'] = df.columns
unique=[]
for i in df.columns:
    unique.append(df[i].nunique())
unique_df['Uniques'] = unique

f, ax = plt.subplots(1,1, figsize=(15,7))

splot = sns.barplot(x=unique_df['Features'], y=unique_df['Uniques'], alpha=0.8)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                   va = 'center', xytext = (0, 9), textcoords = 'offset points')
plt.title('Bar plot for number of unique values in each column',weight='bold', size=15)
plt.ylabel('#Unique values', size=12, weight='bold')
plt.xlabel('Features', size=12, weight='bold')
plt.xticks(rotation=90)
plt.show()


# ##Plot Of Top 15 Locations Of Tweet.

# In[14]:


loc_analysis = pd.DataFrame(df['user_location'].value_counts().sort_values(ascending=False))
loc_analysis = loc_analysis.rename(columns={'user_location':'count'})


# In[15]:


import plotly.graph_objects as go


# In[16]:


data = {
   "values": loc_analysis['count'][:15],
   "labels": loc_analysis.index[:15],
   "domain": {"column": 0},
   "name": "Location Name",
   "hoverinfo":"label+percent+name",
   "hole": .4,
   "type": "pie"
}
layout = go.Layout(title="<b>Ratio on Location</b>", legend=dict(x=0.1, y=1.1, orientation="h"))

data = [data]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title_x=0.5)
fig.show()


# ##Detailed Analysis

# In[17]:


# Make a copy of dataframe before making any changes
tweets = df.copy()


# In[18]:


# Convert date columns to datetime data type from object
tweets['date'] = pd.to_datetime(tweets['date'])
tweets['user_created'] = pd.to_datetime(tweets['user_created'])
tweets['date_ext'] = tweets['date'].dt.date


# In[19]:


# Take care of nulls in location and description
tweets.user_location.fillna('Unknown', inplace=True)
tweets.user_description.fillna('Unknown', inplace=True)
tweets.source.fillna('Unknown', inplace=True)
tweets.hashtags.fillna('None', inplace=True)


# In[20]:


# Verify
tweets.info()


# #Data Preprocessing

# **A) Removing @user**

# In[21]:


# write function for removing @user
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
# create new column with removed @user
df['clean_text'] = np.vectorize(remove_pattern)(df['text'], '@[\w]*')
df.head(2)


# ##REMOVED HTTP AND URLS FROM TWEET

# In[22]:


import re
df['clean_text'] = df['clean_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df.head(3)


# ##**B) Removing Punctuations, Numbers, and Special Characters**

# In[23]:


# remove special characters, numbers, punctuations
df['clean_text'] = df['clean_text'].str.replace('[^a-zA-Z#]+',' ')


# In[24]:


df.head(5)


# ##**C) Removing Short Words**

# In[25]:


# remove short words
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
df.head(2)


# ##**D) Tokenization**

# In[26]:


# create new variable tokenized tweet 
tokenized_tweet = df['clean_text'].apply(lambda x: x.split())
df.head(2)


# ##**E) Stemming**

# In[27]:


from nltk.stem.porter import *
stemmer = PorterStemmer()
# apply stemmer for tokenized_tweet
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
df.head(2)


# In[28]:


# join tokens into one sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
# change df['clean_text'] to tokenized_tweet


# In[29]:


df['clean_text']  = tokenized_tweet
df.head(2)


# ##Story Generation and Visualization from Tweets

# #What are the most common words in the entire dataset?
# 
# *  What are the most common words in the dataset for negative and positive tweets, respectively?
# 
# *  How many hashtags are there in a tweet?
# 
# *  Which trends are associated with my dataset?
# 
# *  Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

# **Understanding the common words used in the tweets: WordCloud**

# In[30]:


df.head(2)


# In[31]:


# create text from all tweets
all_words = ' '.join([text for text in df['clean_text']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# #**Extracting Features from Cleaned Tweets**
# ###Removing Stopwords

# In[32]:


nltk.download('stopwords')


# In[33]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[34]:


df['clean_text'].apply(lambda x: [item for item in x if item not in stop])


# In[35]:


df.head(2)


# ##Check and calculate sentiment of tweets

# In[36]:


#creates a function that determines subjectivity and polarity from the textblob package
def getTextSubjectivity(clean_text):
    return TextBlob(clean_text).sentiment.subjectivity
def getTextPolarity(clean_text):
    return TextBlob(clean_text).sentiment.polarity
#applies these functions to the dataframe
df['Subjectivity'] = df['clean_text'].apply(getTextSubjectivity)
df['Polarity'] = df['clean_text'].apply(getTextPolarity)
#builds a function to calculate and categorize each tweet as Negative, Neutral, and Positive
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"
#creates another column called Score and applies the function to the dataframe
df['Score'] = df['Polarity'].apply(getTextAnalysis)


# In[37]:


#visualizes the data through a bar chart
labels = df.groupby('Score').count().index.values
values = df.groupby('Score').size().values
plt.bar(labels, values, color = ['red', 'blue', 'lime'])
plt.title(label = "Sentiment Analysis - 12/17/2020", 
                  fontsize = '15')
#calculates percentage of positive, negative, and neutral tweets
positive = df[df['Score'] == 'Positive']
print(str(positive.shape[0]/(df.shape[0])*100) + " % of positive tweets")
positive = df[df['Score'] == 'Neutral']
print(str(positive.shape[0]/(df.shape[0])*100) + " % of neutral tweets")
positive = df[df['Score'] == 'Negative']
print(str(positive.shape[0]/(df.shape[0])*100) + " % of negative tweets")


# In[54]:


# Most trended hashtags
top10_hashtags = tweets.hashtags.str.lower().value_counts().nlargest(10)
# initiate the figure with it's size
fig = plt.figure(figsize = (10,5))
plt.barh(top10_hashtags.index, top10_hashtags.values)
plt.xlabel('# of Tweets')
plt.title("Tweets by hashtags", fontsize=16);


# In[66]:


# We are using Compound score to detect the tweet sentiment which is a metric that calculates the sum of
# all the lexicon ratings which have been normalized between 
# -1(most extreme negative) and +1 (most extreme positive)
# positive: (compound score >= 0.05), negative : (compound score <= -0.05), neutral otherwise
get_ipython().system('pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for index, row in tqdm(tweets.iterrows()): #tqdm 
    ss = sid.polarity_scores(row['text'])
    if ss['compound'] >= 0.05 : 
        tweets.at[index,'sentiment'] = "Positive"
    elif ss['compound'] <= - 0.05 : 
        tweets.at[index,'sentiment'] = "Negative"
    else : 
        tweets.at[index,'sentiment'] = "Neutral"


# #Tweets Sentiments Distribution plotted graphically after leveraging NLP

# In[67]:


# Show distribution of tweet sentiments
sentiment_dist = tweets.sentiment.value_counts()

plt.pie(sentiment_dist, labels=sentiment_dist.index, explode= (0.1,0,0),
        colors=['yellowgreen', 'gold', 'lightcoral'],
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Tweets\' Sentiment Distribution \n", fontsize=16, color='Black')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[68]:


# Function to filter top 10 tweets by sentiment
def top10AccountsBySentiment(sentiment):
    df = tweets.query("sentiment==@sentiment")
    top10 = df.groupby(by=["user_name"])['sentiment'].count().sort_values(ascending=False)[:10]
    return(top10)


# In[69]:


# Top 10 tweets by each sentiment
top10_pos = top10AccountsBySentiment("Positive")
top10_neg = top10AccountsBySentiment("Negative")
top10_neu = top10AccountsBySentiment("Neutral")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, squeeze=True, figsize=(16,8))
fig.suptitle('Top 10 Twitter Accounts \n', fontsize=20)

ax1.barh(top10_pos.index, top10_pos.values, color='yellowgreen')
ax1.set_title("\n\n Positive Tweets", fontsize=16)

ax2.barh(top10_neg.index, top10_neg.values, color='lightcoral')
ax2.set_title("\n\n Negative Tweets", fontsize=16)

ax3.barh(top10_neu.index, top10_neu.values, color='gold')
ax3.set_title("\n\n Neutral Tweets", fontsize=16);

fig.tight_layout()
fig.show()


# In[70]:


df.head(1)


# In[71]:


new_df=df[['clean_text','Score']]


# ##Spitting Our Dataset into Training And Testing Dataset ( For Multiclass Classification)

# In[72]:


from sklearn.model_selection import train_test_split

train,valid = train_test_split(new_df,test_size = 0.2,random_state=0,stratify = new_df.Score.values) #stratification means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.
print("train shape : ", train.shape)
print("valid shape : ", valid.shape)


# # Use Of Counter Vectorizer For Multi Class Classification

# In[73]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

X_train = vectorizer.fit_transform(train.clean_text.values)
X_valid = vectorizer.transform(valid.clean_text.values)

y_train = train.Score.values
y_valid = valid.Score.values

print("X_train.shape : ", X_train.shape)
print("X_train.shape : ", X_valid.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_valid.shape)


# ## Naive Bayes Classifier for MULTICLASS Classification

# In[74]:


from sklearn.naive_bayes import MultinomialNB

naiveByes_clf = MultinomialNB()

naiveByes_clf.fit(X_train,y_train)

NB_prediction = naiveByes_clf.predict(X_valid)
NB_accuracy = accuracy_score(y_valid,NB_prediction)
print("training accuracy Score    : ",naiveByes_clf.score(X_train,y_train))
print("Validation accuracy Score : ",NB_accuracy )
print(classification_report(NB_prediction,y_valid))


# #*Thank you! :)*
