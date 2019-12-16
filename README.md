### Twitter_Sentiment_Analysis-US_Airlines


In this project, the tweets [data](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/tree/master/data) of airlines services is used. First, the raw tweets dataset is [pre-processed](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/airlines_sentiment_analysis_exploring.ipynb), then the pre-processed dataset is split to training and test two parts. Before the dataset is fed to model, the text corpus needs to be extracted and transformed to numerical features with word embedding methods such as bag-of-words and TF-IDF. 

![word cloud](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/4.1-text-wordcloud.png)

Since recent machine learning methods such as linear classifier and artificial neural networks are incapable of promising the optimal results, besides [Logistic Regression, Naïve Bayes, KNN, SVM, Decision Tree, Random Forest, AdaBoost, XGBoost and ANN models, LSTM and DCNN deep learning classifiers] (https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/airlines_sentiment_analysis_modelling.ipynb) are also used to identify the optimal model for sentiment analysis in airline services. 

![CNN Achitecture](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/3.6-cnn%20architecture.png)

![LSTM Architecture](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/3.7-lstm%20architecture.png)

The [parameters](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/tree/master/model) of trained CNN and LSTM models can be used into [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning).

In this study, the [descriptive analysis](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/airlines_sentiment_analysis_exploring.ipynb)  and [inferential analysis](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/airlines_sentiment_analysis_exploring.ipynb) are applied to explore the pattern of the tweets dataset. The word frequencies among negative, neutral and positive tweets are different, nevertheless word ‘flight’, ‘get’ and ‘help’ are all appeared in the top 20 common word-list of corpus with all negative, neutral and positive sentiment. 

![Top 20 common words](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/4.2-top20-whole.png)

The biggest amount of tweets is able to be found at [Eastern Time zone](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/4.12-user%20timezone%20count.png), and the most common negative reason is [‘Customer Service Issue’](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/4.16-negative%20reason%20cross%20airline.png). Tweets of the [negative sentiment](https://raw.githubusercontent.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/master/figure/4.17-sentiment%20across%20airline.png) dominate the amount of all tweets corpus. From the overall comparison of introduced machine learning and deep learning models, the optimal model here is <b>CNN</b> with an accuracy <b>0.775</b>, and the worst classifier is <b>KNN</b> model with only <b>0.403</b> accuracy. 

![Model Comparison](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/figure/5.4-model%20comparison.png]

The limitations of this study include the data sample, feature extraction methods and the architecture of classifier models. In the future work, the semantical feature extraction methods can be implemented to capture semantic information, and more advanced models can be applied to promote the prediction accuracy for sentiment analysis. 


In a nutshell, the project offers significant insights to the sentiment analysis for airlines services and provides a feasible direction for the development of sentiment analysis with word embedding methods.

[<h4>Twitter Sentiment with Spark</h4>](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/tree/master/Twitter-Spark)

First put [cleaned data](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/tree/master/Twitter-Spark/code/dataset) to [Hive table](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/Twitter-Spark/code/pyspark-sentiment-analysis-with-hive.ipynb), then use ML models to classify the sentiment on [Spark Framework](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/Twitter-Spark/code/pyspark-sentiment-analysis-with-hive.ipynb).

The Report of twitter sentiment analysis with Spark can download [here](https://github.com/jetrobert/Twitter_Sentiment_Analysis-US_Airlines/blob/master/Twitter-Spark/WQD7007-Project-Sentiment-Analysis-0.2.pdf).
