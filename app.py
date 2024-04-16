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





app = Flask(__name__)

app.config.from_object('config.DevConfig')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

#initializing boto3 client
s3_client = boto3.client('s3', region_name='us-east-2')

UPLOAD_FOLDER= 'uploads' #initializing the uploads folder
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'} #accepts images with only specified extensions 
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER #configuring uploads folder to store uploaded images by the user

#helping function to validate the uploaded image's file extension
def allowed_file(filename): #function to check if the uploaded file's extension is allowed or not
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #security measure to prevent wrong file uploading


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
            



        return render_template('prediction_result.html') 

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/birds')
def birds_page():
    return render_template('birds.html')

if __name__ == '__main__':
    app.run(debug=True)