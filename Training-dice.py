import os

os.chdir("/scratch/yl4217/MS-Lesion-Segmentation/")

config = {}
config["weights_file"] = os.getcwd() + '/model/weight'
config["patch_size"] = (64, 64, 64)  # switch to None to train on the whole image
config["patch_gap"] = 16
config["batch_size"] = 2
config["kfold"] = 5

config["input_shape"] = (1, None, None, None)
config["depth"] = 4 # depth of layers for V/Unet
config["n_base_filters"] = 32
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 10  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["n_epochs"] = 10

from model.data import *
from model.generator import *
from model.dice import *
# from model.model import *

# reference: https://github.com/ellisdg/3DUnetCNN/
import numpy as np

import keras
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
K.set_image_data_format("channels_first")


def unet_model_3d(input_shape, pool_size=(2, 2, 2), initial_learning_rate=1e-4,
                  deconvolution=False, depth=1, n_base_filters=1, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth = 16
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = []

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    # number of labels: 1
    final_convolution = Conv3D(1, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
#     model.compile(optimizer=Adam(lr=initial_learning_rate), loss=keras.losses.binary_crossentropy, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),
                             activation=None, padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def get_callbacks(file_path, file, initial_learning_rate=0.0001, learning_rate_drop=0.5,
                  learning_rate_patience=50, verbosity=1, early_stopping_patience=None):
    
    check_point = ModelCheckpoint(file_path + '/fold' + file + 'weights-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=False)
    csv_log = CSVLogger(file_path + '/training-log.csv', append=True)
    
    # potential problem of recude learning rate: https://github.com/keras-team/keras/issues/10924
    reduce = ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=verbosity)
    if early_stopping_patience:
        early_stop = EarlyStopping(verbose=verbosity, patience=early_stopping_patience)
        return [check_point, csv_log, reduce, early_stop]
    else:
        return [check_point, csv_log, reduce]


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
            initial_learning_rate=0.001, learning_rate_drop=0.5, n_epochs=500,
            learning_rate_patience=20, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    callbacks = get_callbacks(model_file,
                            initial_learning_rate=initial_learning_rate,
                            learning_rate_drop=learning_rate_drop,
                            learning_rate_patience=learning_rate_patience,
                            early_stopping_patience=early_stopping_patience)

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks)

d = Data()
d.load_data(config["patch_size"])

# prepare data for training
train_num, valid_num = d.prekfold(config["patch_size"], config["patch_gap"], config["batch_size"], config["kfold"])
train_generator = DataGenerator(d.data, d.patch_index, d.kfold, d.batch_size, 
                                d.patch_size, d.patch_gap, d.valid_index, True)
valid_generator = DataGenerator(d.data, d.patch_index, d.kfold, d.batch_size, 
                                d.patch_size, d.patch_gap, d.valid_index, False)

result = []
target = []
image = []

print("training selected patch; dice loss")


def train(config, data, train_generator, valid_generator, train_num, valid_num):
#     models = []
    print(train_num, valid_num)
    for i in range(data.kfold):
        print ('-'*100)
        print ("Fold:", i)
        
        train_generator.set_index(i)
        valid_generator.set_index(i)
        
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=config["depth"],
                              n_base_filters=config["n_base_filters"])
        
        # model.load_weights(os.getcwd() + '/model/weight/weights-01-0.02-0428-binary-patch.hdf5')
        
#         print(model.summary())
        
        callbacks = get_callbacks(config["weights_file"], '_dice_'+str(i),
                                initial_learning_rate=config["initial_learning_rate"],
                                learning_rate_drop=config["learning_rate_drop"],
                                learning_rate_patience=config["patience"],
                                early_stopping_patience=config["early_stop"])

        model.fit_generator(generator=train_generator,
                            steps_per_epoch=train_num,
                            epochs=config["n_epochs"],
                            validation_data=valid_generator,
                            validation_steps=valid_num,
                            callbacks=callbacks,
                            workers=2,
                            verbose=1)
        break
train(config, d, train_generator, valid_generator, train_num, valid_num)
