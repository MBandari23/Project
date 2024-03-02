from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import boto3

app = Flask(__name__)

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

@app.route('/predict', methods=['POST']) #upload and prediction route
def predict():
    #if 'bird_image' not in request.files:
        #flash('file part is missing')
        #return redirect(request.url) #pages loads back to MyBirdWat page
    file = request.files['bird_image'] #gets the file from the key "bird_image"
    if file.filename== '':
        flash('No file is selected') #notifies the user that no file is selected

        return redirect(request.url) 
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) 
        print("saved")

        #resizing and preprocessing the image file for the prediction
        imge = Image.open(filepath) #this opens the image file
        imge= imge.resize((229,225)) #Resizing the image as required
        imge.show()
        print("done")
        imge_array = np.expand_dims(np.array(imge) / 255.0, axis=0)
        print(f"Resized image size: {imge.size}")

        #predictions = model.predict(img_array)
        #pred_species = np.argmax(predictions)

        s3_client.upload_file(filepath, 'birdbucket23', 'uploads/' + filename)

        #diplay the image results
        return render_template('prediction_result.html') 



@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/birds')
def birds_page():
    return render_template('birds.html')

if __name__ == '__main__':
    app.run(debug=True)