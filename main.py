
# IMPORTS
import pandas as pd
import spacy
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Importing Dataset
df = pd.read_json("news_dataset.json")
print(df.head())

# Checking whether there are any null values
print(df.columns.isnull())

# Checking all the unique categories
print(df.category.unique())

# Changing Categorical Data to Numerical Data
law = {"SCIENCE" : 0, "BUSINESS" : 1, "CRIME" : 2, "SPORTS" : 3}

for i,k in enumerate(df.category):
    df.category.values[i] = law.get(k)

print(df.head())


# Loading the spacy nlp model
nlp = spacy.load("en_core_web_sm")


# A function for eliminating stop words and punctuations
def preprocess(text):

    doc = nlp(text)

    li = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        
        li.append(token.lemma_)
    
    return " ".join(li)


# Using the function
df["processed_text"] = df.text.apply(preprocess)
# Getting all the vector values
df["vector"] = df["text"].apply(lambda x: nlp(x).vector)


# Splitting Dataset into train, test and x and y
x_train, x_test , y_train, y_test = train_test_split(df.vector.values, df.category.values, test_size=0.2, random_state=2023)


# Stacking the splitted datasets
x_test = np.stack(x_test)
x_train = np.stack(x_train)
y_train = np.stack(y_train)
y_test = np.stack(y_test)


# Checking the data to see inequalities
print(sorted(Counter(y_train).items()))


# Oversampling the data
smt = RandomOverSampler(random_state=70)
x_train, y_train = smt.fit_resample(x_train, y_train)


# Scaling the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# Using the Support Vector Machine (SVM) Model 
model = SVC()
model.fit(x_train,y_train)


# Evaluating the Model 
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

 



