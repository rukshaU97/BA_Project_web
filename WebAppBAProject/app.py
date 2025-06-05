import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load LDA model and vectorizer
with open("lda_model.pkl", "rb") as f:
    lda_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens if word.isalpha() and word not in stop_words
    ]
    return " ".join(filtered_tokens)

# Topic prediction
def predict_topic(text):
    processed_text = preprocess_text(text)
    vector = vectorizer.transform([processed_text])
    topic_distribution = lda_model.transform(vector)
    dominant_topic = int(np.argmax(topic_distribution))

    # Get topic keywords
    topic_words = lda_model.components_[dominant_topic]
    feature_names = vectorizer.get_feature_names_out()
    top_words = [feature_names[i] for i in topic_words.argsort()[-5:][::-1]]

    return dominant_topic, top_words

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Constructive", polarity
    elif polarity < 0:
        return "Critical", polarity
    else:
        return "Neutral", polarity

# Streamlit UI
st.title("Graduate Employability Feedback Analyzer")

questions = [
    "What is your age?",
    "What is your gender?",
    "What is your highest level of education?",
    "What was your field of study?",
    "How many years have passed since your graduation?",
    "What do you think should be improved in Sri Lanka’s higher education system to enhance graduate employability?",
    "What improvements would you suggest for the curriculum to better prepare students for the job market?",
    "What do you think needs to change in the education system or job market to reduce unemployment among bachelor’s degree holders?"
]

responses = []
for q in questions:
    responses.append(st.text_input(q, key=q))

if st.button("Analyze"):
    combined_text = " ".join(responses)
    topic, keywords = predict_topic(combined_text)
    sentiment, polarity = analyze_sentiment(combined_text)

    st.write(f"### Dominant Topic: Topic {topic}")
    st.write("**Key Words:**", ", ".join(keywords))
    st.write(f"**Sentiment:** {sentiment} (Polarity: {polarity:.2f})")
