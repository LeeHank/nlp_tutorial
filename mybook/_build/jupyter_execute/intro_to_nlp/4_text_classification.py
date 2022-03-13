#!/usr/bin/env python
# coding: utf-8

# # Text classification

# In[12]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ## 讀資料 & 切資料

# In[4]:


df = pd.read_csv("data/fake_or_real_news.csv", index_col=0)
df.head()


# In[9]:


# 分 traing, tessting
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size = 0.33, random_state = 53)


# ## 造 feature

# ### bag of words

# In[13]:


count_vectorizer = CountVectorizer(stop_words = "english")
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)


# * 造完 feature 後，我們可以看，現在的 feature 有幾個，以及前 10 個是什麼：

# In[10]:


print(len(count_vectorizer.get_feature_names_out()))
print(count_vectorizer.get_feature_names_out()[:10])


# * 可以看到，造出 56922 個 feature (我猜，就是斷完詞後，共有 56922 個詞). 
# * 前 10 個 feature 看起來頗爛的， 00, 000, ... 這些都是斷詞後的結果，但看來沒啥意義  
# * 回過頭來，等等要拿來 training 用的 X (count_train)，就是每個 instance 在 56922 個 feature 上的 dataframe。 python 很聰明的用 sparse matrix 來存他：

# In[11]:


count_train


# ### tf-idf

# In[14]:


tfidf_vectorizer = TfidfVectorizer(stop_words = "english", max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
tfidf_test = tfidf_vectorizer.transform(X_test.values)


# In[15]:


print(len(tfidf_vectorizer.get_feature_names_out()))
print(tfidf_vectorizer.get_feature_names_out()[:10])


# * 可以看到，feature數是一樣的，因為他只是從詞頻的數值，換成 tf-idf 而已

# In[ ]:


# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))


# ## 建 model

# ### bag of words

# In[18]:


# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels = ['FAKE', 'REAL'])
print(cm)


# ### tfidf

# In[19]:


# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels = ['FAKE', 'REAL'])
print(cm)


# ## Improve model

# * 要 improve model 的方法有很多：  
#   * 調參數. 
#   * 試新 model  
#   * 擴充成更大的 training data
#   * improve text preprocessing

# * 我們先來試試調參數

# In[21]:


# Create the list of alphas: alphas
alphas = np.arange(0, 1, 0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score

# Iterate over the alphas and print the corresponding score
score_list = [train_and_predict(alpha) for alpha in alphas]


# In[28]:


best_index = np.argmax(score_list)
best_alpha = alphas[best_index]
best_score = score_list[best_index]

print(f"best_score: {best_score}")
print(f"best_alpha: {best_alpha}")


# ## Inspecting your model

# In[30]:


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])


# In[31]:


class_labels

