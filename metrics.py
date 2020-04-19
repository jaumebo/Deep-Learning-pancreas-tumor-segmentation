import tensorflow as tf
import tensorflow.keras.backend as K


    
def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-10):
    """ 
    Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return -answer
    
    return loss

def dice_loss():
    return tversky_loss(alpha=0.5,beta=0.5)

def generalized_dice_loss():

    def gen_dice_loss(y_true,y_pred):
        w = K.sum(y_true, axis=(2,3,4))
        w = 1/(w**2+0.000001)
        numerator = y_true*y_pred
        numerator = w*K.sum(numerator, axis=(2,3,4))
        numerator = K.sum(numerator,axis=1)
        denominator = y_true+y_pred
        denominator = w*K.sum(denominator, axis=(2,3,4))
        denominator = K.sum(denominator,axis=1)

        gen_dice_coef = 2*(numerator/denominator)
        gen_dice_coef = tf.reduce_mean(gen_dice_coef)
        return -gen_dice_coef

    return gen_dice_loss

'''
#TEST WITH SMALL TENSORS

import numpy as np
zeros = np.zeros((2,3,2,2,2))
zeros[0,0,:,:,:] = np.ones((zeros.shape[2],zeros.shape[3],zeros.shape[4]))
zeros[1,0,:,:,:] = np.ones((zeros.shape[2],zeros.shape[3],zeros.shape[4]))
tensor1 = tf.convert_to_tensor(zeros, dtype=tf.float32)
zeros = np.zeros((2,3,2,2,2))
zeros[0,0,:,:,:] = np.ones((zeros.shape[2],zeros.shape[3],zeros.shape[4]))
zeros[1,0,:,:,:] = np.ones((zeros.shape[2],zeros.shape[3],zeros.shape[4]))
tensor2 = tf.convert_to_tensor(zeros, dtype=tf.float32)

loss = generalized_dice_loss()

print(loss(tensor1,tensor2))

'''
'''
## TESTING FUNCTIONS

def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [n_labels] + list(data.shape)
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[label_index,:,:,:][data[:,:,:] == labels[label_index]] = 1
        else:
            y[label_index,:][data[:, 0] == (label_index + 1)] = 1
    return y

import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import pylab as plt

#sample_image = "/home/jupyter/ai_postgraduate_project/data/raw_dataset/labelsTr/pancreas_001.nii.gz"
sample_image = "/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/labelsTr/pancreas_001.nii.gz"
second_image = "/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/labelsTr/pancreas_004.nii.gz"

img = nib.load(sample_image)
img = np.array(img.dataobj)
img = get_multi_class_labels(img,3,[0,1,2])
img = img[:,:,:,:107]
img = tf.convert_to_tensor(img, dtype=tf.float32)

img2 = nib.load(second_image)
img2 = np.array(img2.dataobj)
img2 = get_multi_class_labels(img2,3,[0,1,2])
img2 = tf.convert_to_tensor(img2, dtype=tf.float32)

zeros = np.zeros(img.shape)
zeros[0] = np.ones((img.shape[1],img.shape[2],img.shape[3]))
zeros = tf.convert_to_tensor(zeros, dtype=tf.float32)

loss = generalized_dice_loss()

result = loss(img,zeros)

print(result)
'''