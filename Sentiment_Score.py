import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')


# Helper functions
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(word - 3, '?') for word in encoded_review])


def preprocess_text(text):
    max_features = 10000
    words = text.lower().split()
    encoded_review = [
        min(word_index.get(word, 2) + 3, max_features - 1) for word in words
    ]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    score = prediction[0][0]
    sentiment = 'Positive' if score > 0.5 else 'Negative'
    return sentiment, score


# Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below and the model will predict whether it is positive or negative.')

user_input = st.text_area('Movie Review', placeholder='Type your movie review here...')

if st.button('Analyze Sentiment'):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            sentiment, score = predict_sentiment(user_input)

        if sentiment == 'Positive':
            st.success(f'**Sentiment: {sentiment}**')
        else:
            st.error(f'**Sentiment: {sentiment}**')

        st.write(f'**Confidence Score:** {score:.4f}')
        st.progress(float(score))
    else:
        st.warning('Please enter a movie review to analyze.')
