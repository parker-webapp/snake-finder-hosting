from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads
import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from imagedetection import *

# Initializa Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads" 

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)
prev_image = None

# Flask form to upload image
class UploadPhoto(FlaskForm):
	photo = FileField(
		validators=[
			FileAllowed(photos, 'Only images allowed'),
			FileRequired('File field should not be empty')
		]
	)
	submit = SubmitField('Upload')

# Send image to uploads folder
@app.route('/uploads/<filename>')
def get_file(filename):
	return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)

@app.route("/", methods=['GET', 'POST'])
def upload_image():
	form = UploadPhoto()
	snake_name = None
	poisonous = None

	if form.validate_on_submit(): # Check if image was uploaded
		global prev_image
		if prev_image: # Delete the previous uploaded image
			os.remove(prev_image)
		filename = photos.save(form.photo.data)
		file_url = url_for('get_file', filename=filename)
		dir = f"{os.getcwd()}/uploads"
		# if len(os.listdir(f'{dir}')) > 2:
		# 	pass
		print("\n\nLOADER: ", f"{os.getcwd()}{file_url}")
		loader = f"{os.getcwd()}{file_url}" # Add correct path to image
		prev_image = loader
		snake_name,poisonous,description, country, scientific  = image_recognition(loader) # Run image recognition function in imagedetection.py
	else:
		file_url = None
	if snake_name is not None:
		return render_template('index.html', form=form, file_url=file_url, snake_name=snake_name, poisonous=poisonous, description =description, country = country, scientific =scientific)
	else:
		return render_template('index.html', form=form, file_url=file_url)

if __name__ == '__main__':
	app.run(debug=True, port=5001)

