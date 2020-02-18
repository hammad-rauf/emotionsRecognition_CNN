import numpy

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from Images_func import Image_func
from Pickle import Pickle


def cnn_model(feature,label):

    num_classes = 7
    img_rows,img_cols = 48,48

    model = Sequential()

    model.add(BatchNormalization(epsilon=0.001))

    model.add(Conv2D(5,(3,3),padding='same',input_shape=(img_rows,img_cols,1)))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(10,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))    

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))


    model.add(BatchNormalization(epsilon=0.001))
    model.add(Conv2D(15,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Conv2D(20,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))    

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))


    model.add(Conv2D(25,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(30,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))

    
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Conv2D(35,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(40,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))


    model.add(BatchNormalization(epsilon=0.001))
    model.add(Conv2D(45,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(50,(3,3),padding='same'))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))


    model.add(Flatten())
    

    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(21))
    model.add(Activation('elu'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(14))
    model.add(Activation('elu'))
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    checkpoint = ModelCheckpoint('Emotions.h5',
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1)

    earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=9,
                            verbose=1,
                            restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.25,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001)

    callbacks = [earlystop,checkpoint,reduce_lr]

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer = 'adam',
                metrics=['accuracy'])

    epochs=1000
    batch_size=40

    model.fit( feature, label, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks )



if __name__ == "__main__":

    pickle = Pickle()

    ## loading dataset
    label =  pickle.load_label_pickle("label_pickle")
    feature = pickle.load_feature_pickle("feature_pickle")



    # converting numbers btw 0-1 and converting format to float50
    for count in range(len(feature)):
        feature[count] = feature[count]/255.0
        feature[count] = feature[count].astype("float32")


    ## convertin feature and label to numpy array
    feature = numpy.array(feature).reshape(-1,48,48,1)
    label = numpy.array(label)


    cnn_model(feature,label)
