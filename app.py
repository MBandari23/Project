from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import boto3
species_xception = load_model('saved_models/species3_xception.h5')



imagedata = pd.read_csv('data/imagedata.csv', index_col=0)

app = Flask(__name__)

app.config.from_object('config.DevConfig')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

#initializing boto3 client
s3_client = boto3.client('s3', aws_access_key_id='AKIATIIBMARCYEIUQVUG',
    aws_secret_access_key='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',
    region_name='us-east-2'
)

uploads= 'uploads' #initializing the uploads folder
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'} #accepts images with only specified extensions 
app.config['uploads']= uploads #configuring uploads folder to store uploaded images by the user

#helping function to validate the uploaded image's file extension
def allowed_file(filename): #function to check if the uploaded file's extension is allowed or not
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #security measure to prevent wrong file uploading

def model_predict(filepath):
    img = Image.open(filepath)
    img_rs = np.array(img.resize((299,299)))/255
    prediction = species_xception.predict(img_rs.reshape(1,299,299,3))
    return np.round(prediction * 100, 1)[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mybirdwat')
def mybirdwat_page():
    return render_template('mybirdwat.html')

@app.route('/predict', methods=['GET', 'POST']) #upload and prediction route
def predict():
    if request.method == 'POST':
        file = request.files['bird_image'] #gets the file from the key "bird_image"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['uploads'], filename)
            file.save(filepath) 
            print("saved")
            labels = np.unique(np.array(imagedata['species_group'].values))
            prediction = model_predict(filepath)
            top_3 = prediction.argsort()[-1:-4:-1]
            return render_template('predict.html', prediction=prediction, labels=labels, top_3=top_3)
        else:
            flash('An error occurred, try again.')
            return redirect(request.url)  



        



@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/birds')
def birds_page():
    return render_template('birds.html')

if __name__ == '__main__':
    app.run(debug=True)