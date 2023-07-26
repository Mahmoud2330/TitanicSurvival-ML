#!/usr/bin/env python
# coding: utf-8

# # Predict which passengers survived the Titanic shipwreck (Machine Learning Model)

# In[30]:


import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Loading the data and split into features (X) and target variable (y)

# In[34]:


data = pd.read_csv ("train.CSV")
test = pd.read_csv ("test.CSV")
test_ids = test["PassengerId"]


# Creating a cleaning function for the Data

# In[14]:


def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("U", inplace=True)
    return data

data = clean(data)
test = clean(test)


# In[15]:


data.head(5)


# Convert Strings to values

# In[17]:


le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)
    
data.head(5)


# In[23]:


y = data["Survived"]
X = data.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)


# In[31]:


predictions = clf.predict(X_val)
accuracy_score(y_val, predictions)


# In[32]:


submission_preds = clf.predict(test)


# In[35]:


df = pd.DataFrame({"PassengerId":test_ids.values,
                  "Survived": submission_preds,
                  })


# In[36]:


df.to_csv("Submission.csv", index=False)

