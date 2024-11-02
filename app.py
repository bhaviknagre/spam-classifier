import streamlit as st
import pickle
import spacy
import string
from nltk.stem.porter import PorterStemmer

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    # Process the text using SpaCy
    doc = nlp(text.lower())  # Convert text to lowercase

    # Keep only alphanumeric tokens and remove stopwords and punctuation
    y = [ps.stem(token.text) for token in doc if token.is_alpha and not token.is_stop]

    return " ".join(y)

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required model files (vectorizer.pkl, model.pkl) are missing.")

# Streamlit UI
st.title("Email/SMS Spam Classifier")

# Input text area
input_sms = st.text_area("Enter the message", height=150)

# Predict button
if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        st.write("Transformed Message:", transformed_sms)  # Debug: Show transformed message

        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)  # Get prediction probabilities

        # Display result
        if result == 1:
            st.header("Result: Spam")
        else:
            st.header("Result: Not Spam")

        # Debug: Show prediction probabilities
        st.write("Prediction Probabilities:", probabilities)
    else:
        st.warning("Please enter a message to classify.")

