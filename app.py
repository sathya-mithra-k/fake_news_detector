from flask import Flask, request, render_template, jsonify
import pickle
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
vec = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

class_names = ['Fake', 'True']
explainer = LimeTextExplainer(class_names=class_names)

def cleaning(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

def preprocess(text):
    nltk.download('stopwords')
    tokens = word_tokenize(text)  
    stop_words = set(stopwords.words('english'))
    processed_tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(processed_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        
    
        cleaned_text = cleaning(news_text)
        preprocessed_text = preprocess(cleaned_text)
        
        
        text_vec = vec.transform([preprocessed_text])
        
        
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec)[0]

    
        exp = explainer.explain_instance(news_text, predict_proba_wrapper, num_features=10, top_labels=1)
        lime_html = exp.as_html()

        return render_template('result.html', prediction=class_names[prediction], proba=proba, lime_html=lime_html)

def predict_proba_wrapper(texts):
    transformed_texts = vec.transform(texts)
    return model.predict_proba(transformed_texts)

if __name__ == "__main__":
    app.run(debug=True)
