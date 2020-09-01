# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:11:56 2020

@author: aspin.c
"""

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from googletrans import Translator
from langdetect import detect
from flask import Flask , request
#import itertools
#from PyPDF2 import PdfFileReader
import regex as re    
import time
import random
from deeppavlov import build_model, configs
#from werkzeug.utils import secure_filename
#import os
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
START = "./"
app.config["START"] = START


words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('C:/Users/aspin.c/Documents/BU final project chatbot/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
            
a=intents['intents']
ta=[]
la=[]
for i in range(len(a)):
    for j in range(len(a[i]['patterns'])):
        ta.append(a[i]['tag'])
        la.append(a[i]['patterns'][j])
df=pd.DataFrame()
df['sentence']=la
df['label']=ta


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)






model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))),
            ])
            
            


categories=classes
for category in categories:
    model.fit(df['sentence'],df['label'])
    
    
def predict_class(sentence, model):
    pred =model.predict_proba(pd.Series(sentence))
    #print(pred)
    if max(pred[0])>=0.50:
        ma=0.45
        for i in range(len(classes)):
            if pred[:,i][0]>ma:
                ma=pred[:,i][0]
                clas=classes[i]
        #print(ma)          
    else:
        clas="BERT"
    return clas





model_qa = build_model(configs.squad.squad_bert_infer, 
						download = True)
text1=''

def pre(x):
    cor=[]
    
    text = str(x)#.lower()#making all as Lower case
    text = text.replace('\n ', '')
    text = re.sub(r'<.*?>',r'',text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    #text = re.sub('\W', ' ', text) #non words like punct
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)#remove splk char without  numbers
    #text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = re.sub('\s+', ' ', text) 
    cor.append(text) 
    return cor


LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
    'fil': 'Filipino',
    'he': 'Hebrew'
}


@app.route('/',methods = ['GET'])
def home():
        return f"""
                 <!doctype html>
                 <form action="/ans" method="POST" enctype=multipart/form-data>
                <h1>Let Start the Conversation </h1>
                <input name="tex">
                <input type="submit" value="Send">
            </form>"""
    
@app.route('/ans',methods = ['GET','POST'])
def data():
    global text1
    li=[]
    msg=request.form['tex']
    start = time.perf_counter()
    lang = detect(msg)
    translator = Translator()  # initalize the Translator object
    translations = translator.translate(msg, dest='en')  # translate two phrases to Hindi
    msg=translations.text
    if msg=='':
        res ="Please give me more information"        

        
    elif msg.lower()=='quit':
        res="Thanks for chatting"
        return f"""
            <!doctype html>
            <form action="/ans" method="POST" enctype=multipart/form-data>
            <p>User : {msg} </p>
            <p>lang : {lang}</p>
            <p>Azsy : {res} </p>  
                </form>"""
        
    else:
        msg1 = msg.lower()
        tag  = predict_class(msg1, model)
        list_of_intents = intents['intents']
        if(tag=='BERT'):
            if msg1 in text1:
                res ="Already have this information"
                
            elif msg1.endswith('?'):
                    #question=msg1
                    ques=msg1
                    res=model_qa([text1],[ques])[0][0]

                    if res !='':
                        res=res#answer.replace(' ##', '')

                    else :
                        res="Sorry I can't understand it"
            
            else:
                text1=text1+'. '+msg1
                #print(text)
                res="Oh! Thanks for the information"
                
        else:
            for i in list_of_intents:
                if(tag==i['tag']):
                    res=random.choice(i['responses'])
    #trans = translator.translate(res, dest=lang)  # translate two phrases to Hindi
    #res=trans.text
    lang=LANGUAGES[lang]     
    
            
    return f"""
            <!doctype html>
    <form action="/ans" method="POST" enctype=multipart/form-data>
    <p>User     : {msg} </p>
    <p>language : {lang}</p>
    <p>Azsy     : {res} </p>    
    <p>Time taken: {time.perf_counter()-start} </p>
     <input name="tex">
    <input type="submit" value="Send">
            </form>"""

    
if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)   
    
    
