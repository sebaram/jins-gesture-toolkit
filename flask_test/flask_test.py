# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:50:38 2020

@author: kctm
"""

from flask import Flask
from flask import request,render_template
from flaskwebgui import FlaskUI #get the FlaskUI class

import threading, webbrowser


    
app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True



# do your logic as usual in Flask
@app.route("/")
@app.route('/index')
def my_form():
    return render_template('form.html')


# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['text']
#     processed_text = text.upper()
#     print("get text: "+processed_text)
#     return render_template('form.html', name=processed_text)

@app.route('/', methods=['GET', 'POST'])
def my_form_post2():
    print(request.form['input_name'])
    text = request.form['input_name']
    processed_text = text.upper()
    print("get text: "+processed_text)
    
    name_of_slider = request.form["input_number_of_gesture"]
    print("slider: "+name_of_slider)
    
    return render_template('form.html', name=processed_text)

# call the 'run' method
if __name__ == '__main__':
    # Feed it the flask app instance 
    # ui = FlaskUI(app)
    # ui.run()
    

    url = "http://127.0.0.1:5000"

    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    app.run()