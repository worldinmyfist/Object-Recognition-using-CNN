from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy

for i in range(20):
    # Initialising the CNN
    classifier = Sequential()
    # Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) #(64,64,3) is (pixel,pixel,3), it can be manually altered
    # Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Flattening
    classifier.add(Flatten())
    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the CNN to the images
    train_datagen = ImageDataGenerator(rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('dataset/training_set',#make a folder dataset containing training images in training_set folder
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('dataset/test_set',#add your test cases in dataset folder in test_set
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'categorical')
    classifier.fit_generator(training_set,
            steps_per_epoch = 64,
            epochs = 2,#can be increased for accuracy but compilation time increases
            validation_data = test_set,
            validation_steps = 2000)
    # Making new predictions
    item0, item1 = 0, 0
    for i in range(150):#number of test images : 150
        x = 'dataset/test_set/Item-0/img' + str(i + 1) + '.jpg'
        test_image = image.load_img(x, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            item1 += 1
        else:
            item0 += 1
    print('No. of items of type 0: ', item0, '.', 'No. of items of type 1: ', item1, '.')
    print()

