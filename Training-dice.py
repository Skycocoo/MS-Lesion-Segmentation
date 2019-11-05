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
from model.modelDice import *

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

        # print(model.summary())

        callbacks = get_callbacks(config["weights_file"], str(i)+'_dice_',
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
