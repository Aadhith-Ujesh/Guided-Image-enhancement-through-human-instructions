import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from PIL import Image

model = load_model('latentSpaceModel.h5')

BATCH_SIZE = 256; EPOCHS = 10
train_datagen = ImageDataGenerator(rescale=1./255)
train_batches = train_datagen.flow_from_directory('D:/FYP/dataset/resized_dataset/',
        target_size=(64,64), shuffle=True, class_mode='input', batch_size=BATCH_SIZE)


images = next(iter(train_batches))[0]


for i in range(20264):

    orig = images[i,:,:,:].reshape((-1,64,64,3))

    img_orig = Image.fromarray((255 * orig).astype('uint8').reshape((64,64,3)))
    
    
    latent_img = model.predict(orig)
    
    #print(latent_img)
    np.save('dataset/latent/' + str(i+1) + '.npy', latent_img)
