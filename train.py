from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)

from data import get_balanced_train_and_validation_datasets
from homemade_unet import unet_model_3d
from metrics import dice_loss, generalized_dice_loss, tversky_loss
from utils.callback_functions import ShowImagesCallback

#Paths must be changed according to the actual working directory
path_images = '/content/gdrive/My Drive/Task07_Pancreas/imagesTr/'
path_labels = '/content/gdrive/My Drive/Task07_Pancreas/labelsTr/'
path_models_best = '/content/gdrive/My Drive/models/best/'
path_models_final = '/content/gdrive/My Drive/models/final/'
path_runs = '/content/gdrive/My Drive/runs/'


input_shape = (1,None,None,None)
learning_rate=0.001
loss_func = generalized_dice_loss()

model = unet_model_3d(input_shape = input_shape, 
                      n_labels = 3, 
                      loss = loss_func, 
                      initial_learning_rate=learning_rate)

batch_size = 2

train_dataset, validation_dataset, validation_images = get_balanced_train_and_validation_datasets(0.2,
                                                                                         path_images,
                                                                                         path_labels,
                                                                                         patch_shape=(128,128,64),
                                                                                         validation_shape=(256,256,64),
                                                                                         mask=True,
                                                                                         repetitions=2,
                                                                                         proportion_background=1.5)
train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(3)
validation_dataset = validation_dataset.shuffle(50).batch(1).prefetch(2)

name = '150_gendice_15background_alltransf'

model_checkpoint_best = ModelCheckpoint(filepath=path_models_best + name + '_model.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

model_checkpoint_epoch = ModelCheckpoint(filepath=path_models_final + name + '_model.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   period=1)

model_checkpoint_epoch_colab = ModelCheckpoint(filepath=name + '_model.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   period=1)

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', 
                                      patience=10, 
                                      verbose=1)

tensorboard_callback = TensorBoard(log_dir=path_runs + name,
                                   write_grads=True,
                                   histogram_freq=1)

file_writer = tf.summary.create_file_writer(path_runs + name)
show_results_callback = ShowImagesCallback(train_dataset,validation_dataset,file_writer,num_images=4)

callbacks = [show_results_callback,model_checkpoint_epoch,model_checkpoint_epoch_colab,tensorboard_callback,model_checkpoint_best,reduce_lr_plateau]#,early_stop]

epochs = 1000

try:
    model.fit(train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks = callbacks)
except KeyboardInterrupt:
    model.save_weights(path_models_final + name + '_model_stop.h5')
    print('Output model saved')