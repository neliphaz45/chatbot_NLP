# import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from textblob import Word
import numpy as np

from tfidf_model import QuestionAnswer


app = Flask(__name__)
model = joblib.load(open('question_answer.pkl', 'rb'))
  
@app.route('/')      # my default homepage
def home():
    return render_template('index.html')     # where your input needs to be


@app.route('/predict/', methods=['GET','POST'])
def predict():

    if request.method == "POST":
        #get form data
        question = request.form.get('question')
        #call preprocessInputData and pass inputs
        preproc_question = preprocessInputData(question)
        answer =  model.get_answer_percontext([preproc_question])
        #pass prediction to template
        return render_template('index.html', prediction_text = 'The answer to this question: \n "{}" \n is: "{}"'.format(question, answer))
    pass

def preprocessInputData(input_question):
    #pre processing steps like lower case, stemming and lemmatization
    input_question = input_question.lower()
    # stop = stopwords.words('english')
    # input_question = " ".join(x for x in input_question.split() if x not in stop)
    # st = PorterStemmer()
    # input_question = " ".join ([st.stem(word) for word in input_question.split()])
    # input_question = " ".join ([Word(word).lemmatize() for word in input_question.split()])
    return input_question

if __name__ == "__main__":
    app.run(debug=True)