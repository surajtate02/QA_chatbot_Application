import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np

# Scrape and clean data from Wikipedia
def scrape_wikipedia(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract text from paragraphs
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    
    # Basic text cleaning
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Train a simple QA model
def train_qa_model(data):
    questions = data['question']
    answers = data['answer']
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    
    model = LogisticRegression()
    model.fit(X, answers)
    
    return vectorizer, model

# Predict answer
def predict_answer(question, vectorizer, model):
    X = vectorizer.transform([question])
    prediction = model.predict(X)
    return prediction[0]

# Example usage
if __name__ == '__main__':
    url = 'https://en.wikipedia.org/wiki/Chatbot'
    text = scrape_wikipedia(url)
    
    # Create a simple dataset
    data = {
        'question': [
            'What is a chatbot?',
            'Who invented the Turing test?',
            'What is natural language processing?',
            'What is machine learning?',
            'What is AI?',
            'What is a conversational agent?',
            'What is a virtual assistant?',
            'What is the purpose of a chatbot?',
            'How do chatbots work?',
            'What are the applications of chatbots?'
        ],
        'answer': [
            'A chatbot is a software application used to conduct an on-line chat conversation via text or text-to-speech.',
            'Alan Turing.',
            'Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.',
            'Machine learning is a type of artificial intelligence that allows software applications to become more accurate in predicting outcomes without being explicitly programmed to do so.',
            'AI stands for artificial intelligence, the simulation of human intelligence in machines.',
            'A conversational agent is a software program that engages in conversation with a human.',
            'A virtual assistant is a software agent that can perform tasks or services for an individual based on commands or questions.',
            'The purpose of a chatbot is to simulate a conversation with a human.',
            'Chatbots work by using pre-programmed responses or machine learning algorithms to respond to user input.',
            'Chatbots can be used in customer service, information acquisition, and as personal assistants.'
        ]
    }
    
    df = pd.DataFrame(data)
    vectorizer, model = train_qa_model(df)
    
    # Test the model
    test_question = 'What is a chatbot?'
    print(predict_answer(test_question, vectorizer, model))
