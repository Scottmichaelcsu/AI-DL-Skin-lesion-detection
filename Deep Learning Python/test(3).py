#Split data into lesion type files
import pandas as pd
import os
import shutil

# Dump all images into a folder and specify the path:
data_dir = 'C:/Users/grayj/Desktop/DATASET/BothImages'

# Path to destination directory where we want subfolders
dest_dir = 'C:/Users/grayj/Desktop/DATASET/Subfolders'

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv('C:/Users/grayj/Desktop/DATASET/HAM10000_metadata.csv')
print(skin_df2['dx'].value_counts())

label=skin_df2['dx'].unique().tolist()  #Extract labels into a list
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df2[skin_df2['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
    label_images=[]    

#Now we are ready to work with images in subfolders
'''   
### FOR Keras datagen ##################################
#flow_from_directory Method
#useful when the images are sorted and placed in there respective class/label folders
#identifies classes automatically from the folder name. 
# create a data generator
    
from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt

#Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()

# define training directory that contains subfolders
train_dir = os.getcwd() + '/Ai-DLdata/data/'
#USe flow_from_directory
train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical',
                                         batch_size=16,  #16 images at a time
                                         target_size=(32,32))  #Resize images

#We can check images for a single batch.
x, y = next(train_data_keras)
#View each image
for i in range (0,15):
    image = x[i].astype(int)
    plt.imshow(image)
    plt.show()
    '''