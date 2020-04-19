import itertools
import os
import random

import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage import exposure as ex

from utils.patches import compute_patch_indices, get_patch_from_3d_data
from utils.transformation import apply_transformations

def equalize(image_array):
    '''Function used to apply equalization to the images'''
    image_equalized = ex.equalize_hist(image_array)
    return image_equalized

def has_labels(img):
    '''Function used to determine wether if a patch has pancreas or not'''
    return bool((1 in img)*(2 in img))

def window_image_min(img, img_min=-100, img_max=250):
    '''Function used to convert pixel intensity values with min and max pixel value parameters'''
    window_image = img.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def normalize_image(img):
    """
    Take an image and normalize its values
    """
    if (img.max()-img.min())!=0:
        norm_img = (img - img.min())/(img.max()-img.min())
    else:
        norm_img = (img - img.min())
    return norm_img

def get_multi_class_labels(data, n_labels, labels=[0,1,2]):
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
        y[label_index,:,:,:][data[:,:,:] == labels[label_index]] = 1
    return y

def resize_image(img, input_shape, output_shape):

    '''
    Given an image, resize it to the chosen output_shape

    Params:
    img: input image in nifti format
    input_shape: input shape of the image
    output_shape: desired output shape
    '''

    initial_size_x = int(input_shape[0])
    initial_size_y = int(input_shape[1])
    initial_size_z = int(input_shape[2])

    new_size_x = int(output_shape[0])
    new_size_y = int(output_shape[1])
    new_size_z = int(output_shape[2])

    initial_data = img.get_data()

    delta_x = initial_size_x/new_size_x
    delta_y = initial_size_y/new_size_y
    delta_z = initial_size_z/new_size_z

    new_data = np.full((new_size_x,new_size_y,new_size_z),1024)

    for x, y, z in itertools.product(range(new_size_x),
                                    range(new_size_y),
                                    range(new_size_z)):
        new_data[x][y][z] = initial_data[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]

    img_resized = nib.Nifti1Image(new_data, np.eye(4))

    return img_resized

def path_to_np(path,
               list_img,
               img_num,
               resize,
               resize_shape,
               expand=True,
               mask=False,
               label=False):

    '''
    Given the path of an image, return the numpy array of it.

    Params:
    path: directory path where the images are
    list_img: list of images in path
    img_num: which image from list_img wants to be processed
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    expand: if we want to expand first dimention to send through neural network
            example: input -> (X,Y,Z), output -> (1,X,Y,Z)
    '''

    current_image = list_img[img_num].decode('utf-8')  
    #print("------- Image to be processed:")
    path_img = os.path.join(path, current_image)
    #print(path_img)
    
    img = nib.load(path_img)

    img = np.array(img.dataobj)

    if resize:
        img = resize_image(img,img.shape,resize_shape)

    img = img[27:411,113:422,:]

    if not label:
        if mask:
            img = window_image_min(img)
        img = normalize_image(img)
        img = equalize(img)

    if expand:
        img = np.expand_dims(img,axis=0)
    
    return img

def create_dataset(list_images,
                   path_images,
                   path_targets,
                   resize=False,
                   resize_shape=(0,0,0),
                   mask=False):

    '''
    Return a TensorFlow dataset object for whole images.

    Params:
    list_images: list of images that we want the dataset have
    path_images: path where the images are
    path_targets: path where the labels/targets are
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    '''

    def data_iterator(stop,
                      path_images,
                      path_targets,
                      list_images,
                      resize,
                      resize_shape,
                      mask):

        '''
        Iterator inside create_dataset to yield the images.
        '''

        path_images = path_images.decode('utf-8')
        path_targets = path_targets.decode('utf-8')
        i=0
        while i<stop:

            img = path_to_np(path_images,list_images,i,resize,resize_shape,mask=mask)

            img = normalize_image(img)

            img = equalize(img)

            label = path_to_np(path_targets,list_images,i,resize,resize_shape,expand=False,mask=mask,label=True)
            
            label = get_multi_class_labels(label,3,[0,1,2])

            yield (img,label)
            i+=1

    if resize:
        out_shape_im = (1,resize_shape[0],resize_shape[1],resize_shape[2])
        out_shape_lb = (3,resize_shape[0],resize_shape[1],resize_shape[2])
    else:
        out_shape_im = (1,512,512,95)
        out_shape_lb = (3,512,512,95)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[len(list_images),path_images,
                                                  path_targets,list_images,resize,resize_shape,mask],
                                            output_shapes=(out_shape_im,out_shape_lb),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    return dataset


def image_to_patches(img,patch_size):

    '''
    img: image as numpy array
    patch_shape: tuple of patch shape (for example: (126,126,45))

    returns: list of images (patches) created of the chosen size
    '''

    indices = compute_patch_indices(img.shape,patch_size)

    images = []

    for index in indices:
        images.append(get_patch_from_3d_data(img,patch_size,index))
    
    return images


def next_patch(img,lbl,patch_shape,indices,index,expand):

    '''
    Given an image, return the following patch that has not been processed yet.

    Params:
    img: image in numpy array
    patch_shape: desired shape of the patch
    indices: localizations of the different patches we are going to get. These
             are calculated with function compute_patch_indices.
    index: which index on indices are we using in this iteration.
    expand: whether to expand dimension of image True/False

    Returns: 
    img: actual patch of the image, numpy array
    indices: vector of indices
    index: next index to be processed
    finished: True/False whether if we have finished with the current image patches or not
    '''
    
    if not indices is None:
        img = get_patch_from_3d_data(data=img,
                                     patch_shape=patch_shape,
                                     patch_index=indices[index])
        lbl = get_patch_from_3d_data(data=lbl,
                                     patch_shape=patch_shape,
                                     patch_index=indices[index])
        index+=1        
    else:
        indices = compute_patch_indices(image_shape=img.shape,patch_size=patch_shape)
        img = get_patch_from_3d_data(data=img,
                                     patch_shape=patch_shape,
                                     patch_index=indices[0])
        lbl = get_patch_from_3d_data(data=lbl,
                                     patch_shape=patch_shape,
                                     patch_index=indices[0])
        index = 1
    
    if index==len(indices):
        finished=True
    else:
        finished=False
    
    if expand:
        img = np.expand_dims(img,axis=0)
            
    return img, lbl, indices, index, finished


def patches_dataset(list_images,
                    path_images,
                    path_targets,
                    patch_shape,
                    resize=False,
                    resize_shape=(0,0,0),
                    mask=False):

    '''
    Return a TensorFlow dataset object for patched images.

    Params:
    list_images: list of images that we want the dataset have
    path_images: path where the images are
    path_targets: path where the labels/targets are
    patch_shape: shape of patches that want to be processed
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    '''

    def data_iterator(path_images,
                      path_targets,
                      patch_shape,
                      list_images,
                      resize,
                      resize_shape,
                      mask):

        path_images = path_images.decode('utf-8')
        path_targets = path_targets.decode('utf-8')

        cont = True
        i=0
        indices = None
        index = 0
        finished = False
        img_num = 0

        while cont:
            
            if indices is None:
                try:
                    big_img = path_to_np(path=path_images,
                                        list_img=list_images,
                                        img_num=img_num,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        expand=False,
                                        mask=mask)
                                        
                    big_label = path_to_np(path=path_targets,
                                        list_img=list_images,
                                        img_num=img_num,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        expand=False,
                                        mask=mask,
                                        label=True)
                except:
                    img_num+=1
                    indices=None
                    continue

            img, label, indices, index, finished = next_patch(img=big_img,
                                                                lbl=big_label,
                                                                patch_shape=patch_shape,
                                                                indices=indices,
                                                                index=index,
                                                                expand=True)

            
            if finished:
                img_num+=1
                indices = None
                index = 0
            
            if img_num==len(list_images):
                cont = False

            label = get_multi_class_labels(label,3,[0,1,2])

            yield (img,label)
            i+=1

    out_shape_im = [1,patch_shape[0],patch_shape[1],patch_shape[2]]
    out_shape_im = tuple(out_shape_im)

    out_shape_lb = [3,patch_shape[0],patch_shape[1],patch_shape[2]]
    out_shape_lb = tuple(out_shape_lb)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[path_images,path_targets,
                                                  patch_shape,list_images,resize,resize_shape,mask],
                                            output_shapes=(out_shape_im,out_shape_lb),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    #dataset = dataset.batch(batch_size)

    return dataset


def split_images(list_images,split,seed):

    ''' 
    Given a list of images, return a random split for train and validation

    Params:
    list_images: list of images to be splitted
    split: split ratio (example: split=0.2 --> train 80%, validation 20%)
    seed: which seed for random
    '''

    random.seed(seed)

    num_images = len(list_images)
    num_validation = int(np.round(num_images*split))
    
    train_images = list_images
    random.shuffle(train_images)
    validation_images = []

    for _ in range(num_validation):
        validation_images.append(train_images.pop())
    
    print("Number of images for training: " + str(len(train_images)))
    print("Number of images for validation: " + str(len(validation_images)))

    return train_images, validation_images


def get_train_and_validation_datasets(
        split,
        path_images,
        path_targets,
        subsample=None,
        patch=True,
        patch_shape=(216,216,64),
        resize=False,
        resize_shape=(0,0,0),
        seed=123,
        mask=False):

    '''
    Return TensorFlow datasets for train and validate:

    Params:
    split: split ratio (example: split=0.2 --> train 80%, validation 20%)
    path_images: path where the images are
    path_targets: path where the labels/targets are
    subsample: None if all pictures to be analyzed or a number (int) of pictures
    patch: whether to resize the image or not (True/False)
    patch_shape: shape of patches that want to be processed
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    seed: seed for random
    mask: whether to use classical computer vision mask or not (True/False)
    '''
    #list_images = os.listdir(path_images)
    
    list_images = []
    for file in os.listdir(path_images):
        if file.endswith(".nii.gz"):
            list_images.append(file)

    if subsample is not None:
        list_images = list_images[:subsample]
        print("Subsampling with images: " + str(list_images))


    train_images, validation_images = split_images(list_images=list_images,
                                                   split=split,
                                                   seed=seed)

    if patch:
        train_dataset = patches_dataset(list_images=train_images,
                                        path_images=path_images,
                                        path_targets=path_targets,
                                        patch_shape=patch_shape,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        mask=mask)

        validation_dataset = patches_dataset(list_images=validation_images,
                                             path_images=path_images,
                                             path_targets=path_targets,
                                             patch_shape=patch_shape,
                                             resize=resize,
                                             resize_shape=resize_shape,
                                             mask=mask)
    else:
        train_dataset = create_dataset(list_images=train_images,
                                       path_images=path_images,
                                       path_targets=path_targets,
                                       resize=resize,
                                       resize_shape=resize_shape,
                                       mask=mask)

        validation_dataset = create_dataset(list_images=validation_images,
                                            path_images=path_images,
                                            path_targets=path_targets,
                                            resize=resize,
                                            resize_shape=resize_shape,
                                            mask=mask)
    
    return train_dataset, validation_dataset, validation_images


def get_balanced_train_and_validation_datasets(
        split,
        path_images,
        path_targets,
        subsample=None,
        patch_shape=(128,128,64),
        validation_shape=(256,256,128),
        resize=False,
        resize_shape=(0,0,0),
        seed=123,
        mask=False,
        repetitions=3,
        proportion_background=1.5):

    '''
    Return TensorFlow datasets for train and validate:

    Params:
    split: split ratio (example: split=0.2 --> train 80%, validation 20%)
    path_images: path where the images are
    path_targets: path where the labels/targets are
    subsample: None if all pictures to be analyzed or a number (int) of pictures
    patch: whether to resize the image or not (True/False)
    patch_shape: shape of patches that want to be processed
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    seed: seed for random
    mask: whether to use classical computer vision mask or not (True/False)
    repetitions: number of times we want to repeat a patch with pancreas (applying transformations)
    '''
    #list_images = os.listdir(path_images)
    
    list_images = []
    for file in os.listdir(path_images):
        if file.endswith(".nii.gz"):
            list_images.append(file)

    if subsample is not None:
        list_images = list_images[:subsample]
        print("Subsampling with images: " + str(list_images))


    train_images, validation_images = split_images(list_images=list_images,
                                                   split=split,
                                                   seed=seed)


    train_dataset = patches_balanced_dataset(list_images=train_images,
                                            path_images=path_images,
                                            path_targets=path_targets,
                                            patch_shape=patch_shape,
                                            resize=resize,
                                            resize_shape=resize_shape,
                                            mask=mask,
                                            repetitions=repetitions,
                                            proportion_background=proportion_background)

    validation_dataset = patches_dataset(list_images=validation_images,
                                        path_images=path_images,
                                        path_targets=path_targets,
                                        patch_shape=validation_shape,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        mask=mask)
    
            
    return train_dataset, validation_dataset, validation_images



def patches_balanced_dataset(list_images,
                            path_images,
                            path_targets,
                            patch_shape,
                            resize=False,
                            resize_shape=(0,0,0),
                            mask=True,
                            repetitions=3,
                            proportion_background=1.5):

    '''
    Return a TensorFlow dataset object for patched images.

    Params:
    list_images: list of images that we want the dataset have
    path_images: path where the images are
    path_targets: path where the labels/targets are
    patch_shape: shape of patches that want to be processed
    resize: whether to resize the image or not (True/False)
    resize_shape: if resize==True, the desired output shape of the resize
    mask: whether to apply HU mask or not
    repetitions: number of repetitions (using data augmentation) used for each pancreas patch
    proportion_background: number of background patches for each pancreas patch sent to the model
    '''

    def data_iterator(path_images,
                      path_targets,
                      patch_shape,
                      list_images,
                      resize,
                      resize_shape,
                      mask,
                      repetitions,
                      proportion_background):

        path_images = path_images.decode('utf-8')
        path_targets = path_targets.decode('utf-8')

        cont = True
        i=0
        new_img = True
        index = 0
        finished = False
        img_num = 0

        while cont:
            
            if new_img:
                new_img = False
                try:
                    big_img = path_to_np(path=path_images,
                                        list_img=list_images,
                                        img_num=img_num,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        expand=False,
                                        mask=mask)

                    big_label = path_to_np(path=path_targets,
                                        list_img=list_images,
                                        img_num=img_num,
                                        resize=resize,
                                        resize_shape=resize_shape,
                                        expand=False,
                                        mask=mask,
                                        label=True)
                except:
                    img_num += 1
                    new_img = True
                    continue

                indices, full_indices = get_chosen_indices(big_label,patch_shape,repetitions=repetitions,proportion_background=proportion_background)
                
            try:
                patch_tupla, index, finished = next_patch_balanced(big_img = big_img,
                                                                big_lbl = big_label,
                                                                patch_shape = patch_shape,
                                                                index=index,
                                                                indices=indices,
                                                                full_indices = full_indices)
            except:
                print("Error with image " + str(list_images[img_num]))
                img_num += 1
                new_img = True
                index = 0
                if img_num==len(list_images):
                    cont = False 
                continue
            
            if finished:
                img_num+=1
                new_img = True
                index = 0
            
            if img_num==len(list_images):
                cont = False

            img = patch_tupla[0]
            img = np.expand_dims(img, axis=0)
            label = get_multi_class_labels(patch_tupla[1],3,[0,1,2])

            yield (img,label)
            i+=1

    out_shape_im = [1,patch_shape[0],patch_shape[1],patch_shape[2]]
    out_shape_im = tuple(out_shape_im)

    out_shape_lb = [3,patch_shape[0],patch_shape[1],patch_shape[2]]
    out_shape_lb = tuple(out_shape_lb)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[path_images,path_targets,
                                                  patch_shape,list_images,resize,resize_shape,mask,repetitions,proportion_background],
                                            output_shapes=(out_shape_im,out_shape_lb),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    #dataset = dataset.batch(batch_size)

    return dataset


def next_patch_balanced(big_img, big_lbl, patch_shape, index, indices, full_indices):
    '''Given the input image and its label, return the next patch that the input pipeline has to return'''
    finished = False

    img = get_patch_from_3d_data(big_img,patch_shape,full_indices[indices[index]])
    lbl = get_patch_from_3d_data(big_lbl,patch_shape,full_indices[indices[index]])
    
    if index==(len(indices)-1):
        finished = True
    else:
        index+=1

    patch_tupla = random_transform_couple((img,lbl))

    return patch_tupla, index, finished


def get_chosen_indices(lbl,patch_shape,repetitions,proportion_background):
    '''Return the corner indices for each of the selected patches for each image'''

    full_indices = compute_patch_indices(lbl.shape,patch_shape)

    index_distribution = {'background':[],'target':[]}

    index_num=0

    for index in full_indices:
         
        patch_lbl = get_patch_from_3d_data(lbl, patch_shape, index)

        if has_labels(patch_lbl):
            index_distribution['target'].append(index_num)
        else:
            index_distribution['background'].append(index_num)
        
        index_num += 1

    background_indices = list(np.random.choice(list(index_distribution['background']),
                                                size=repetitions*int(np.round(proportion_background * len(index_distribution['target']))),
                                                replace=True))
    
    indices = background_indices
    for _ in range(repetitions):
        indices = indices  + index_distribution['target']
    
    np.random.shuffle(indices)

    return indices, full_indices



def random_transform_couple(couple):
    '''Applies random transformations for data augmentation'''

    couple = apply_transformations(image = couple[0],
                                target = couple[1],
                                apply_flip_axis_x = True,
                                apply_flip_axis_y = True,
                                apply_flip_axis_z = True,
                                apply_gaussian_offset = False,
                                apply_gaussian_noise = False,
                                apply_elastic_transfor = False,
                                sigma_gaussian_offset = None,
                                sigma_gaussian_noise = None,
                                alpha_elastic = None,
                                sigma_elastic = None)
    return couple
