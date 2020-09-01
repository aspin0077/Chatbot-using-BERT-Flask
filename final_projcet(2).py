# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:11:56 2020

@author: aspin.c
"""


from flask import Flask , request

import time
import itertools
from PyPDF2 import PdfFileReader
import regex as re
import chardet
#import deeppavlov
from deeppavlov import build_model, configs
from werkzeug.utils import secure_filename
import os
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
START = "./"
app.config["START"] = START


model_qa = build_model(configs.squad.squad_bert_infer, 
						download = True)
text1=[]
   
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



@app.route('/',methods = ['GET','POST'])
def data():
    return f"""
    <!doctype html>
    <form action="/data" method="POST" enctype=multipart/form-data>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <input type=file name=file>
    <input type=submit value=Upload>
            </form>"""


@app.route('/data',methods = ['GET','POST'])
def ind():
    global text1
    text1=[]
    file = request.files['file']
    filename = secure_filename(file.filename)
    new_path = os.path.abspath(filename)
    print(filename)
    print(new_path)
    if filename is not None: 
        if filename.endswith('.txt'):
            with open(new_path, 'rb') as f:
                # Join binary lines for specified number of lines
                rawdata = b''.join([f.readline() for _ in range(100)])
            encode=chardet.detect(rawdata)['encoding']
            nw=open(new_path,encoding=str(encode)) 
            content = nw.read()  
            text1=pre(content)
            print(text1)  
        elif filename.endswith('.pdf'):
            pdf =PdfFileReader(open(new_path, "rb"))
            for page in pdf.pages:
                text1.append(pre(page.extractText()))
            text1=list(itertools.chain.from_iterable(text1))
            text1=' '.join(text1)
            text1=list([text1])
            print(text1)
            
    return f"""
                 <!doctype html>
                <form action="/ans" method="POST">
                <input name="tex">
                <input type="submit" value="Ask">
                </form>"""
            
            
@app.route("/ans", methods=['POST'])
def echo(): 
    global text1
    msg=request.form['tex']
    start = time.perf_counter()
    ques=msg
    ans =''
    
    ans=model_qa(text1,[ques])[0][0]
    if ans==None:
        ans='There is no relevent data for this question'
    print(ans)
   
    print('Time:',time.perf_counter()-start)
    return f"""
                <!doctype html>
                <p>Ques: {msg} </p>
                <p>Ans: {ans} </p>
                <p>Time taken: {time.perf_counter()-start} </p>
                <form action="/ans" method="POST">
                <input name="tex">
                <input type="submit" value="Ask">
                </form>
                """
    
    
if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)   
    
    
