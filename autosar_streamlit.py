import re
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle



# Read data from CSV
df = pd.read_csv("dataset.csv")

# Preprocess the text data
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing Punctuation and Numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

    
df['Requirement'] = df['Requirement'].apply(preprocess_text)
print(df)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import numpy as np

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['Class'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Requirement'], encoded_labels, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = max(len(seq) for seq in X_train + X_test)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Define model
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

model = Sequential()
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
model.add(embedding_layer)
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred_classes)
print("\n")
print(classification_report(y_test, y_pred_classes))



# Streamlit app
st.title('AUTOSAR Requirements Classification App')

user_input = st.text_input("Enter a sentence to classify:")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    # Tokenize and pad the new requirement
    new_requirement_seq = tokenizer.texts_to_sequences([preprocessed_input])
    new_requirement_padded = pad_sequences(new_requirement_seq, maxlen=max_len, padding='post')

    # Predict label for the new requirement
    predicted_label_probabilities = model.predict(new_requirement_padded)
    predicted_label_index = np.argmax(predicted_label_probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    st.write(f'Predicted Class: {predicted_label}')