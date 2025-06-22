import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

twt = TweetTokenizer()
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = twt.tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("E-mail/SMS Spam Detection")

input_sms = st.text_area("Enter the message here")

if st.button("Predict"):
    
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")