# reference: https://github.com/ellisdg/3DUnetCNN/
# import numpy as np

# need to use keras backend instead of numpy
from keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1.):
#     y_true = K.cast(y_true, 'float64')
#     y_pred = K.cast(y_pred, 'float64')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersec = K.sum(y_true_f * y_pred_f)
    return (2. * intersec + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
#     y_true_f = np.array(y_true).flatten()
#     y_pred_f = np.array(y_pred).flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     # print(intersection, np.sum(y_true_f), np.sum(y_pred_f))
#     # tensorflow computation graph: will not configure print as one of the graph, unless using tf.Print()
#     return (2.*intersection+smooth) / (np.sum(y_true_f)+np.sum(y_pred_f)+smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# # compute dice score by label
# def label_wise_dice_coefficient(y_true, y_pred, label_index):
#     return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])
#
# def get_label_dice_coefficient_function(label_index):
#     f = partial(label_wise_dice_coefficient, label_index=label_index)
#     f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
#     return f
