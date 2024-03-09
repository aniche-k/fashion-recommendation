from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
# from livelossplot import PlotLossesKerasTF
# from IPython.display import SVG, Image
# from livelossplot.tf_keras import PlotLossesCallback
# from livelossplot import PlotLossesKerasTF
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from numpy import asarray
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import os,glob
import json


def get_recommendations_from_json(occasion,uname,selected_path = "None"):
    metadata_path = os.path.join("users", uname, "metadata.json")
    with open(metadata_path, "r") as json_file:
        metadata_json = json.load(json_file)
    if selected_path == "None":
        return metadata_json["default"][occasion]
    else:
        return metadata_json[selected_path][occasion]

def metadata_exists(occasion,uname,selected_path = "None"):
    metadata_file_path = os.path.join("users",uname,"metadata.json")
    if not os.path.exists(metadata_file_path):
        with open(metadata_file_path,"w") as metadata_file:
            json.dump({
                "default": {
                    "casual": [],
                    "party": [],
                    "office": []
                },
                "new_items": []
            }, metadata_file)
        return False
    with open(metadata_file_path,"rb") as metadata_file:
        metadata_json = json.load(metadata_file)
    if selected_path != "None":
        if selected_path not in metadata_json.keys():
            metadata_json[selected_path] = {"casual": [], "party": [], "office": []}
            with open(metadata_file_path, "w") as metadata_file:
                json.dump(metadata_json, metadata_file)
            return False
        elif len(metadata_json[selected_path][occasion]) == 0:
            return False
        else:
            return True
    elif len(metadata_json["default"][occasion]) != 0:
        return True
    else:
        return False

def recommend_default_new(occasion,uname,selected_type = "None",selected_path = "None"):
    print(occasion)
    path_pants = []
    path_shirts = []
    path_shoes = []
    recommendation_type_key = selected_path if selected_path != "None" else "default"
    occasion_dict = {
        "casual":0,
        "party":0,
        "office":0
    }
    model=tf.keras.models.load_model('./models/fashion-reccomendation1.h5',compile=False)
    model.compile()
    occasion_dict[occasion] = 1
    return_imgs = lambda item_type: glob.glob(os.path.join("users",uname,item_type,"**","*.jpg"),recursive=True)
    if selected_type == "pants":
        pant_imgs = [selected_path]
    else:
        pant_imgs = return_imgs("pants")
    if selected_type == "shirt" or selected_type == "tshirt":
        shirt_imgs = [selected_path]
    else:
        shirt_imgs = return_imgs("shirt") + return_imgs("t-shirt")
    if selected_type == "shoes":
        shoe_imgs = [selected_path]
    else:
        shoe_imgs = return_imgs("shoes")
    # empty_dict = {uname+"\default":[]}
    empty_dict = {recommendation_type_key:{
        "casual":[],
        "party":[],
        "office":[]
    }}
    value_list = []
    for shirt in shirt_imgs:
        for pant in pant_imgs:
            for shoe in shoe_imgs:
                shirt_img = np.resize(cv2.imread(shirt),(1,128,128,3))
                pant_img = np.resize(cv2.imread(pant),(1,128,128,3))
                shoe_img = np.resize(cv2.imread(shoe),(1,128,128,3))
                output = model.predict([shirt_img,pant_img,shoe_img,np.array([list(occasion_dict.values())])])
                empty_dict[recommendation_type_key][occasion].append(
                    {
                        "*".join([shirt,pant,shoe]) : output[0][0]
                    }
                )
                print(shirt)
                print(output[0][0])
    empty_dict[recommendation_type_key][occasion] = [list(item.keys())[0] for item in sorted(empty_dict[recommendation_type_key][occasion],key=lambda item:list(item.values())[0],reverse=True)]
    print(empty_dict)
    return empty_dict

# recommend_default_new("party","Mahesh")
# metadata_exists("casual","hello")

# img1=cv2.imread(r'"C:\Users\anees\loginflask\users\hello\shirt\black\1.jpg"')
# img2=cv2.imread(r'"C:\Users\anees\loginflask\users\hello\pants\green\1.jpg"')
# img3=cv2.imread(r'"C:\Users\anees\loginflask\users\hello\shoes\black\2.jpg"')

# print(img1)