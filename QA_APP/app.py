from flask import Flask, render_template, request
from qa_bot import scrape_wikipedia, train_qa_model, predict_answer
import pandas as pd

app = Flask(__name__)

# Load and prepare data
url = 'https://en.wikipedia.org/wiki/Chatbot'
text = scrape_wikipedia(url)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_question = request.form['question']
        answer = predict_answer(user_question, vectorizer, model)
        return render_template('index.html', question=user_question, answer=answer)
    return render_template('index.html', question='', answer='')

if __name__ == '__main__':
    app.run(debug=True)
