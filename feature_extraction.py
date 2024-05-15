import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image

def extract_features(directory_path, image_size=(224, 224)):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=Flatten()(base_model.layers[-1].output))
    features = []

    for root, dirs, files in os.walk(directory_path):
        for file in tqdm(files):
            img_path = os.path.join(root, file)
            
            try:
                # Open the image using PIL to check if it's a valid image file
                Image.open(img_path)
                
                img = image.load_img(img_path, target_size=image_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model.predict(x)
                features.append(feature.flatten())
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.vstack(features)


data_directory = 'new_dataset'
features = extract_features(data_directory)

feature_df = pd.DataFrame(features)
feature_df.to_csv('image_features.csv', index=False)

