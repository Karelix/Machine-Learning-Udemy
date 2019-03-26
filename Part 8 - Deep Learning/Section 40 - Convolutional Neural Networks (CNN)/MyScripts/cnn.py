# Convolutional Newral Networks

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Part 1 - Convolution
classifier.add(Convolution2D(32,3,data_format='channels_last',input_shape=(64,64,3),activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second confolutional layer
classifier.add(Convolution2D(32,3,data_format='channels_last',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128,activation='relu')) # Hidden
classifier.add(Dense(units=1,activation='sigmoid')) # Output

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the CNN to all the images

# Generating new images from the dataset to avoid overfittin
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../dataset/training_set',
        target_size=(64, 64), #dimensions expected from CNN
        batch_size=32,      #update weights after every 32 
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '../dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32, # number of entries in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32) # number of entries in testing set

# Finding if my dog is a dog :p
import numpy as np
from keras.preprocessing import image
img = image.load_img('sparky.jpg',target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
result = classifier.predict(img)
training_set.class_indices
