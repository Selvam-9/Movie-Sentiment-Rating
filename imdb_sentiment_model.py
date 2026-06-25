from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

max_feature =100000 #vocabsize

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=max_feature)


sample_review = X_train
sample_label = y_train

reversed_word_index = { value:key for key,value in imdb.get_word_index().items()}

decode_words = " ".join([reversed_word_index.get(i-3,'?') for i in sample_review[0]])
max_len = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)



model = Sequential()
model.add(Embedding(max_feature,128,))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
earlystop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


#model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=.2,callbacks=[earlystop])

#model.save('simple_rnn_imdb.h5')
import streamlit as st
import tensorflow as tf

@st.cache_resource  # This ensures the model is loaded only once
def load_model():
    return tf.keras.models.load_model('my_sentiment_model.h5')

model = load_model()

# Now use 'model' for predictions
# user_input = st.text_input("Enter a review")
# prediction = model.predict(...)
