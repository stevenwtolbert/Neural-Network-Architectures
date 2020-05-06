import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

class AlexNet(Sequential):
    def __init__(self, input_shape,num_classes):
        super().__init__()
        
        #Convolutional Layer 1
        self.add(Conv2D(filters = 96,
                        kernel_size=(11,11),
                        strides = 4,
                        padding = 'valid', 
                        activation = 'relu',
                        input_shape = input_shape, 
                        kernel_initializer = 'he_normal'))
        #Normalize and Pool
        self.add(BatchNormalization()) #LRN would be applied here classically 
        self.add(MaxPooling2D(pool_size=(3,3), 
                              strides = (2,2),
                              padding = 'valid',
                              data_format = None))
        
        #Convolutional Layer 2
        self.add(Conv2D(filters = 256,
                        kernel_size=(5,5),
                        strides = 1,
                        activation = 'relu',
                        padding = 'same',
                        kernel_initializer = 'he_normal'))
        
        #Normalize and Pool
        self.add(BatchNormalization()) #LRN would be applied here classically 
        self.add(MaxPooling2D(pool_size=(3,3), 
                              strides = (2,2),
                              padding = 'valid'))

        #Convolutional Layers 3-4, without normalizing or pooling.
        self.add(Conv2D(filters = 384,
                        kernel_size=(3,3),
                        strides = 1,
                        activation = 'relu',
                        padding = 'same',
                        kernel_initializer = 'he_normal'))
        
        self.add(Conv2D(filters = 384,
                        kernel_size=(3,3),
                        strides = 1,
                        activation = 'relu',
                        padding = 'same',
                        kernel_initializer = 'he_normal'))
        #Convolutional Layer 5
        self.add(Conv2D(filters = 256,
                        kernel_size=(3,3),
                        strides = 1,
                        activation = 'relu',
                        padding = 'same',
                        kernel_initializer = 'he_normal'))
        
        #Normalize and Pool
        self.add(MaxPooling2D( pool_size=(3,3), 
                               strides = (2,2),
                               padding = 'valid'))
        #Layer 6-8 FCNs
        self.add(Flatten())
        self.add(Dense(4096, activation = 'relu'))
        self.add(Dropout(0.4))
        self.add(Dense(4096, activation = 'relu'))
        self.add(Dropout(0.4))
        self.add(Dense(1000, activation = 'relu'))
        self.add(Dense(num_classes, activation= 'softmax'))

        self.compile(optimizer = tf.keras.optimizers.Adam(1E-3),
                     loss = 'categorical_crossentropy',
                     metrics = ['accuracy'])

model = AlexNet((227, 227, 3), num_classes = 1000)
