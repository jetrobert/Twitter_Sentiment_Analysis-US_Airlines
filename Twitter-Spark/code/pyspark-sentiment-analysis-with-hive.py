
# coding: utf-8

# ### Libraries Dependency

# In[1]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext


# ### Creating Spark context

# In[2]:


SparkContext.setSystemProperty("spark.executor.memory", "4g")
sc = SparkContext('local[1]')
hc = HiveContext(sc)


# In[3]:


sc._conf.getAll()


# ### Read a table from Hive

# In[4]:


hc.sql('use project')
df = hc.sql('select * from tweet_orc where line_number is not null')
df.show(10)


# In[5]:


df.printSchema()


# In[6]:


df.select("line_number").show(10)


# In[7]:


type(df)


# ### drop nan

# In[8]:


#df = df.dropna()
df.count()


# ### split dataset

# In[9]:


(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed = 2000)


# ### Logistic Regression with TFIDF

# In[10]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[13]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
#minDocFreq: remove sparse terms
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) 
label_stringIdx = StringIndexer(inputCol = "label", outputCol = "class")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
test_df = pipelineFit.transform(test_set)
train_df.show(5)


# In[15]:


lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)

predictions = lrModel.transform(val_df)
# evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
# evaluator.evaluate(predictions)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
print("valication accuracy: ", accuracy)
predictions_test = lrModel.transform(test_df)
accuracy_test = predictions_test.filter(predictions_test.label == predictions_test.prediction).count() / float(test_set.count())
print("test accuracy: ", accuracy_test)


# In[17]:


#evaluator.getMetricName()


# ### Logistic Regression with CountVectorizer and IDF

# In[11]:


from pyspark.ml.feature import CountVectorizer


# In[12]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
#minDocFreq: remove sparse terms
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) 
label_stringIdx = StringIndexer(inputCol = "label", outputCol = "class")
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])
pipelineFit = pipeline.fit(train_set)

#train_df = pipelineFit.transform(train_set)
#val_df = pipelineFit.transform(val_set)
#test_df = pipelineFit.transform(test_set)
#train_df.show(5)


# In[16]:


predictions_val = pipelineFit.transform(val_set)
accuracy = predictions_val.filter(predictions_val.label == predictions_val.prediction).count() / float(val_set.count())
print("Validation Accuracy Score: {0:.4f}".format(accuracy))
#roc_auc = evaluator.evaluate(predictions)
#print "ROC-AUC: {0:.4f}".format(roc_auc)

predictions_t = pipelineFit.transform(test_set)
accuracy_test = predictions_t.filter(predictions_t.label == predictions_t.prediction).count() / float(test_set.count())
print("Test Accuracy Score: {0:.4f}".format(accuracy_test))


# ### Logisitic Regression with N-gram

# In[18]:


from pyspark.ml.feature import NGram


# In[31]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
ngram = NGram(n=1, inputCol="words", outputCol="n_gram")
hashtf = HashingTF(numFeatures=2**16,inputCol="n_gram", outputCol="tf")
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) 
label_stringIdx = StringIndexer(inputCol = "label", outputCol = "class")
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, ngram, hashtf, idf, label_stringIdx, lr])
pipelineFit = pipeline.fit(train_set)


# In[34]:


predictions_val = pipelineFit.transform(val_set)
accuracy = predictions_val.filter(predictions_val.label == predictions_val.prediction).count() / float(val_set.count())
print("Validation Accuracy Score: {0:.4f}".format(accuracy))

predictions_t = pipelineFit.transform(test_set)
accuracy_test = predictions_t.filter(predictions_t.label == predictions_t.prediction).count() / float(test_set.count())
print("Test Accuracy Score: {0:.4f}".format(accuracy_test))

