
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")


# In[13]:


csv = './dataset/clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()


# ### Word Cloud

# In[3]:


neg_tweets = my_df[my_df.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')


# In[4]:


from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Some of the big words can be interpreted quite neutral, such as "today","now", etc.
# I can see some of the words in smaller size make sense to be in negative tweets,
# such as "damn","ugh","miss","bad", etc.
# But there is "love" in a rather big size, so I wanted to see what is happening.

# OK, even though the tweets contain the word "love", in these cases it is negative sentiment because the tweet has mixed emotions like "love" but "miss". Or sometimes used in a sarcastic way.

# In[5]:


for t in neg_tweets.text[:200]:
    if 'love' in t:
        print t


# In[6]:


pos_tweets = my_df[my_df.target == 1]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')


# In[7]:


wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Again I see some neutral words in big size, "today","now",
# but words like "haha", "love", "awesome" also stand out.

# Interestingly, the word "work" was quite big in the negative word cloud, but also quite big in the positive word cloud.
# It might imply that many people express negative sentiment towards work, but also many people are positive about works.
