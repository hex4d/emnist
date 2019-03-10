from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras import models
from keras import layers
from keras import optimizers
from keras.engine.topology import Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras import utils

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# paramater
batch_size=32
SEED=1470
input_size = 224

base_dir='data'
train_dir=os.path.join(base_dir, 'train')
model_base_dir = 'models/base'

def category_to_char(category_number):
    return chr(category_number + ord('a') - 1)

def show_images(gen):
    x, y = next(gen)
    for i in range(len(x)):
        plt.subplot(4, 8, i + 1)
        plt.title(chr(y[i].argmax() + ord('a') - 1))
        plt.imshow(x[i].reshape((28, 28)))
    plt.show()

def show_test(test_gen, model):
    x, y = next(test_gen)
    pred = model.predict(x).argmax(axis=1)
    for i in range(len(x)):
        plt.subplot(4, 8, i + 1)
        if category_to_char(pred[i]) != category_to_char(y[i].argmax()):
            plt.title('×' + category_to_char(pred[i]) + ' - ' + category_to_char(y[i].argmax()))
        else:
            plt.title(category_to_char(pred[i]) + ' - ' + category_to_char(y[i].argmax()))
        plt.imshow(x[i].reshape((28, 28)))
    plt.show()


def split_with_class_count(df, validation_split=0.05):
    classes = df[0].unique()
    val_classes = df[0].unique()
    train_df = pd.DataFrame(columns=df.columns)
    train_df = pd.concat([train_df, df])
    validation_df = pd.DataFrame(columns=df.columns)
    for val_class in val_classes:
        class_df = df[df[0] == val_class]
        validation = class_df.sample(frac=validation_split, random_state=SEED)
        validation_df = pd.concat([validation_df, validation]) 
        train_df = train_df.drop(validation.index)
    train_df = train_df.reset_index()
    validation_df = validation_df.reset_index()
    print('train', len(train_df), 'validation', len(validation_df))
    return train_df, validation_df, classes.tolist()

def get_generator(df, params):
  x = df.iloc[:,2:].values.astype(float)
  y = df.iloc[:,1].values.astype(int)
  x = x.reshape(x.shape[0], 28, 28, 1)
  x = np.transpose(x, (0, 2, 1, 3))
  y = utils.to_categorical(y)
  datagen = ImageDataGenerator(**params)
  gen = datagen.flow(
    x,
    y,
    batch_size=32,
    seed=SEED,
  )
  return gen

def load_data():
  df = pd.read_csv(os.path.join(base_dir, 'emnist/emnist-letters-train.csv'), header=None)
  train_df, validation_df, classes = split_with_class_count(df, validation_split=0.1)
  test_df = pd.read_csv(os.path.join(base_dir, 'emnist/emnist-letters-test.csv'), header=None)
  test_df, _, _ = split_with_class_count(test_df, validation_split=0.0)
  train_gen = get_generator(train_df, {
      'rescale':1./255,
      'rotation_range':30,
      'width_shift_range':0.1,
      'height_shift_range':0.1,
      'shear_range':0.1
  })
  val_gen = get_generator(validation_df, {
      'rescale':1./255,
  })
  test_gen = get_generator(test_df, {
      'rescale':1./255,
  })
  return train_gen, val_gen, test_gen

class ModelV1():
    def __init__(self):
        self.name = 'v1'
    def get_model(self):
        main_input = Input(shape=(28, 28, 1))
        x = layers.Conv2D(28, (2, 2), activation='relu', padding='same')(main_input)
        x = layers.Conv2D(28, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(14, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(14, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(7, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(7, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(3, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(3, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(27, activation='softmax')(x)
        model = models.Model(inputs=[main_input], outputs=[x])
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    def model_path(self, base_dir):
        return os.path.join(base_dir, self.name) 

class ModelV2():
    def __init__(self):
        self.name = 'v2'
    def get_model(self):
        main_input = Input(shape=(28, 28, 1))
        x = layers.Conv2D(28, (2, 2), activation='relu', padding='same')(main_input)
        x = layers.Conv2D(28, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(14, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(14, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(14, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(7, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(7, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(7, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(27, activation='softmax')(x)
        model = models.Model(inputs=[main_input], outputs=[x])
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    def model_path(self, base_dir):
        return os.path.join(base_dir, self.name) 

class ModelV3():
    def __init__(self):
        self.name = 'v3'
    def get_model(self):
        main_input = Input(shape=(28, 28, 1))
        x = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(main_input)
        x = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(27, activation='softmax')(x)
        model = models.Model(inputs=[main_input], outputs=[x])
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    def model_path(self, base_dir):
        return os.path.join(base_dir, self.name) 

if __name__ == '__main__':
    
    epochs = 100
    train_gen, val_gen, test_gen = load_data()
    model_wrapper = ModelV3()
    model = model_wrapper.get_model()
    model.summary()
    os.makedirs(model_wrapper.model_path(model_base_dir), exist_ok=True)
    model_checkpoint_path = os.path.join(model_wrapper.model_path(model_base_dir), '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5' )
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=os.path.join(model_wrapper.model_path(model_base_dir)))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.1, factor=0.25, min_lr=0.0002, verbose=1)

    # iteration = 1
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=50,
        callbacks=[model_checkpoint, tensor_board, reduce_lr],
        initial_epoch=0
        # initial_epoch=(iteration-1)*epochs
    )
