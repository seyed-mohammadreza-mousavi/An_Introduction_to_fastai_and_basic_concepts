##########################################BINARY CLASSIFICATION##########################################
# First you have to install fastbook
# For google.colab we have:
!pip install -Uqq fastbook

# Importing essential libraries
import fastbook
from fastbook import *
from fastai.vision.all import *

# Downloading Dogs vs Cats dataset
dest = '/content/DataML'
path = untar_data (URLs.PETS, data=dest)/'images' 

# We know Cats pictures filenames starts with a capital so we define the is_cat() function as follows:
def is_cat(x): return x[0].isupper() 

# Creating DataLoader
dls = ImageDataLoaders.from_name_func(path=path, fnames=get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))

# Define the model similar to resnet34
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# Fine-tunning the model
learn.fine_tune(1)

# Now we wanna check the accuracy on our model:

# First we upload a picture of dog or cat of our own:
uploader = widgets.FileUpload()
uploader

# Now we predict whether it is a cat or dog
img = PILImage.create(uploader.data[0])
pred,_,_ = learn.predict(img)
if pred == 'True': 
    print(f'Prediction: Cat')
else: 
    print(f'Prediction: Dog')
