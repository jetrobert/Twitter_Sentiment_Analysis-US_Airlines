
# coding: utf-8

# ### Dataset

# In[4]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")


# First, columns names have been assigned to each column.

# In[5]:


cols = ['sentiment','id','date','query_string','user','text']


# In[6]:


df = pd.read_csv("./dataset/trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols)


# In[7]:


df.head()


# In[9]:


df.sentiment.value_counts()


# In[11]:


df.drop(['id','date','query_string','user'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df[df.sentiment == 0].head(10)


# In[14]:


df[df.sentiment == 4].head(10)


# In order for the computation, I mapped the class value of 4(positive) to 1.

# In[17]:


df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})


# In[18]:


df.sentiment.value_counts()


# ## Data Preparation

# The order of the cleaning is 
# 1. Souping
# 2. BOM removing
# 3. url address('http:'pattern), twitter ID removing
# 4. url address('www.'pattern) removing
# 5. lower-case
# 6. negation handling
# 7. removing numbers and special characters
# 8. tokenizing and joining

# In[34]:


import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()


# In[39]:


#pat1 = r'@[A-Za-z0-9]+'
#pat2 = r'https?://[A-Za-z0-9./]+'
#combined_pat = r'|'.join((pat1, pat2))

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


# In[40]:


testing = df.text[:100]


# In[41]:


test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))


# In[42]:


test_result


# In[43]:


nums = [0,400000,800000,1200000,1600000]


# In[44]:


get_ipython().run_cell_magic(u'time', u'', u'print "Cleaning and parsing the tweets...\\n"\nclean_tweet_texts = []\nfor i in xrange(nums[0],nums[1]):\n    if( (i+1)%10000 == 0 ):\n        print "Tweets %d of %d has been processed" % ( i+1, nums[1] )                                                                    \n    clean_tweet_texts.append(tweet_cleaner(df[\'text\'][i]))')


# In[45]:


len(clean_tweet_texts)


# In[46]:


get_ipython().run_cell_magic(u'time', u'', u'print "Cleaning and parsing the tweets...\\n"\nfor i in xrange(nums[1],nums[2]):\n    if( (i+1)%10000 == 0 ):\n        print "Tweets %d of %d has been processed" % ( i+1, nums[2] )                                                                    \n    clean_tweet_texts.append(tweet_cleaner(df[\'text\'][i]))')


# In[47]:


len(clean_tweet_texts)


# In[48]:


get_ipython().run_cell_magic(u'time', u'', u'print "Cleaning and parsing the tweets...\\n"\nfor i in xrange(nums[2],nums[3]):\n    if( (i+1)%10000 == 0 ):\n        print "Tweets %d of %d has been processed" % ( i+1, nums[3] )                                                                    \n    clean_tweet_texts.append(tweet_cleaner(df[\'text\'][i]))')


# In[49]:


len(clean_tweet_texts)


# In[50]:


get_ipython().run_cell_magic(u'time', u'', u'print "Cleaning and parsing the tweets...\\n"\nfor i in xrange(nums[3],nums[4]):\n    if( (i+1)%10000 == 0 ):\n        print "Tweets %d of %d has been processed" % ( i+1, nums[4] )                                                                    \n    clean_tweet_texts.append(tweet_cleaner(df[\'text\'][i]))')


# In[51]:


len(clean_tweet_texts)


# ### Saving cleaned data as csv

# In[52]:


clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()


# In[53]:


clean_df.to_csv('./dataset/clean_tweet.csv',encoding='utf-8')


# In[56]:


csv = './dataset/clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()

