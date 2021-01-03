from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.models import model_from_json

classifier = Sequential()
classifier.add(Conv2D(36, (3, 3), input_shape = (30, 30, 3),strides=1, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(72, (3, 3),strides = 2, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 30, activation = 'softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.1,zoom_range = 0.1)
training_set = train_datagen.flow_from_directory('dataset\\training set',batch_size = 55,target_size=(30,30))

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('dataset\\test set',batch_size = 20,target_size=(30,30))

from keras.callbacks import ModelCheckpoint
filepath = "BestWeights14_new.h5"
checkPoint = ModelCheckpoint(filepath, monitor= "val_acc", save_best_only= True, mode ='max')
callbacks_list = [checkPoint]

classifier.fit_generator(training_set,epochs = 10,validation_data = test_set, validation_steps = 35, callbacks=callbacks_list)

model_json = classifier.to_json()
with open("model14_new.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
classifier.save_weights("model14_new.h5")
print("Model saved to disk")
