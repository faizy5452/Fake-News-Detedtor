import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("train.csv")
df=df.fillna(' ')
df['content']=df['author']+' '+df['title']
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['content']=df['content'].apply(stemming)
x=df['content']
y=df['label']
vector=TfidfVectorizer()
vector.fit(x)
x=vector.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=1)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


st.title("Fake News Detector")
text_input=st.text_input('Enter News Article Here ')

def prediction(text):
    input_data=vector.transform([text])
    prediction=model.predict(input_data)
    return prediction[0]

if text_input:
    pred=prediction(text_input)
    if pred ==1:
        st.write("The News is Fake")
    else:
        st.write("The News is Real")


