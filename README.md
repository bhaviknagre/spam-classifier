# Project Title

*Spam Classifier ML Project*

## Overview
This project presents a Spam Classifier using Machine Learning techniques. The application is built with Streamlit, enabling users to classify messages as spam or not in real time. The project employs natural language processing (NLP) techniques to preprocess input data and a machine learning model to make predictions.

## Data Overview
The dataset consists of labeled SMS messages categorized into 'spam' and 'ham' (non-spam). The data is processed to extract relevant features, which are then used to train the classification model. Key features include:
- Message content
- Labels (spam/ham)

## Recommendation Workflow
1. **Data Collection**: Gathered SMS messages for training and testing.
2. **Data Preprocessing**: Cleaned and normalized text data using techniques such as lowercasing, tokenization, stop word removal, and stemming.
3. **Feature Extraction**: Transformed the processed text into numerical format using TF-IDF vectorization.
4. **Model Training**: Trained the Multinomial Naive Bayes classifier using the processed dataset.
5. **Deployment**: Created a user-friendly interface with Streamlit for real-time predictions.

## Key Features
- User-friendly web interface for message input.
- Real-time spam classification with result display.
- Detailed prediction probabilities for transparency.
- Robust text preprocessing techniques for improved accuracy.

## Parameters
- **TF-IDF Vectorizer**: Max features set to 3000 for effective feature extraction.
- **Naive Bayes Classifier**: Utilized Multinomial Naive Bayes suitable for discrete count data.

## Key Insights
- The classifier effectively identifies spam messages based on the content provided.
- Preprocessing steps significantly enhance the model's performance by reducing noise in the data.


# Usage
- Run the Streamlit application:
           - streamlit run app.py
- Enter a message in the input box and click "Predict" to see the classification result.


# Files Descriptions
- app.py: Main Streamlit application file.
- vectorizer.pkl: Pickled TF-IDF vectorizer used for feature extraction.
- modelmnb.pkl: Pickled Multinomial Naive Bayes model for spam classification.
- requirements.txt: List of Python packages required for the project.


# Technologies Used
- Python: Programming language used for developing the model and application.
- Streamlit: Framework for creating web applications.
- scikit-learn: Library for machine learning algorithms and utilities.
- spaCy: NLP library for text processing.
- NLTK: Library for natural language processing tasks such as stemming.

# Link to Web App
https://spam-classifier-kgjfnt6wqkxchjcbodcdhp.streamlit.app/

# Visuals

# Results and Recommendations
The spam classifier has shown promising results, accurately classifying messages with high precision and recall. It is recommended to further enhance the model by:

  - Incorporating more extensive datasets for training.
  - Exploring advanced NLP techniques like word embeddings.


# Conclusion
This Spam Classifier project demonstrates the practical application of machine learning and natural language processing. The integration of a streamlit interface allows for easy interaction,making it accessible for users who need to identify spam messages quickly.   
