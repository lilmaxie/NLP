import numpy as np
import  tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st

# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# load the model
model = load_model('RNN/imdb_rnn_model.h5')

# Helper function to decode reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# function to preprocess user input
def preprocess_text(text):
    # Convert the text to lowercase
    words = text.lower().split()
    # encode the words to integers
    encoded = [word_index.get(word, 2) + 3 for word in words]
    # pad the sequence to the same length as the training data
    padded_review = sequence.pad_sequences([encoded], maxlen=500)
    return padded_review

# predict function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    # make prediction
    prediction = model.predict(preprocessed_input)
    # convert prediction to sentiment
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]


# streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative')

# user input
user_review = st.text_area("Movie Review", "Type your review here...")

if st.button("Classify"):
    preprocess_input = preprocess_text(user_review)
    
    # make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    # display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
    
else:
    st.write("Click the button to classify the review.")
    st.write("Example: 'This movie was fantastic! I loved it.'")
    st.write("Example: 'This movie was terrible. I hated it.'")