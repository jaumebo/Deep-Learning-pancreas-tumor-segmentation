import numpy as np
import tensorflow as tf
import nibabel as nib

def has_labels(img):
    return bool((1 in img)*(2 in img))

def get_slice(img):

    z_slices = []
    for z in range(img.shape[-1]):
        if has_labels(img[:,:,z]):
            z_slices.append(z)

    z = int(np.median(z_slices))

    return img[:,:,z], z

def paint_img(name,im_slice,writer):
    im_slice = np.expand_dims(im_slice,axis=-1)
    im_slice = (255*(im_slice/2)).astype(np.uint8)
    with writer.as_default():
        tf.summary.image(name, tf.convert_to_tensor(im_slice), step=0)
    pass

class ShowPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self,train_dataset,val_dataset,file_writer,loss,num_images=1,print_loss=False):
        self.loss = loss
        self.print_loss = print_loss
        self.file_writer=file_writer

        count = 0
        self.img_patches_train=[]
        self.ground_truth_train=[]
        for batch in train_dataset:
            for i in range(batch[0].shape[0]):
                x = batch[0][i,:,:,:,:]
                y = batch[1][i,:,:,:,:]
                self.ground_truth_train.append(np.expand_dims(y,axis=0))
                y = np.argmax(y,axis=0)
                if has_labels(y):
                    count+=1
                    slice_y, z_slice = get_slice(y)
                    slice_y = np.expand_dims(slice_y,axis=0)
                    self.img_patches_train.append((x,z_slice))
                    paint_img(name="Ground Truth train " + str(count),im_slice = slice_y,writer=self.file_writer)
                    if count == num_images:
                        break
            if count == num_images:
                break
        
        count=0
        self.img_patches_val=[]
        self.ground_truth_val=[]
        for batch in val_dataset:
            for i in range(batch[0].shape[0]):
                x = batch[0][i,:,:,:,:]
                y = batch[1][i,:,:,:,:]
                self.ground_truth_val.append(np.expand_dims(y,axis=0))
                y = np.argmax(y,axis=0)
                if has_labels(y):
                    count+=1
                    slice_y, z_slice = get_slice(y)
                    slice_y = np.expand_dims(slice_y,axis=0)
                    self.img_patches_val.append((x,z_slice))
                    paint_img(name="Ground Truth validation " + str(count),im_slice = slice_y,writer=self.file_writer)
                    if count == num_images:
                        break
            if count == num_images:
                break

    def on_epoch_end(self, epoch, logs=None):
        
        count=0
        for (img, z_slice) in self.img_patches_train:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction train " + str(count),im_slice=res_slice,writer=self.file_writer)
            loss_res = self.loss(self.ground_truth_train[count-1],np.expand_dims(res_model,axis=0))
            if self.print_loss:
                print("Loss train " + str(count) + ":" + str(loss_res))
        
        count=0
        for (img, z_slice) in self.img_patches_val:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction validation " + str(count),im_slice=res_slice,writer=self.file_writer)
            loss_res = self.loss(self.ground_truth_val[count-1],np.expand_dims(res_model,axis=0))
            if self.print_loss:
                print("Loss validation " + str(count) + ":" + str(loss_res))

class ShowImagesCallback(tf.keras.callbacks.Callback):
    def __init__(self,train_dataset,val_dataset,file_writer,num_images=1):
        self.file_writer=file_writer

        count_pancreas = 0
        count_background = 0
        self.img_patches_train_pancreas=[]
        self.img_patches_train_background=[]
        for batch in train_dataset:
            for i in range(batch[0].shape[0]):
                x = batch[0][i,:,:,:,:]
                y = batch[1][i,:,:,:,:]
                y = np.argmax(y,axis=0)
                if has_labels(y) and count_pancreas!=num_images:
                    count_pancreas+=1
                    slice_y, z_slice = get_slice(y)
                    slice_y = np.expand_dims(slice_y,axis=0)
                    slice_x = x[:,:,:,z_slice]
                    self.img_patches_train_pancreas.append((x,z_slice))
                    paint_img(name="Ground Truth image train pancreas" + str(count_pancreas),im_slice = slice_x,writer=self.file_writer)
                    paint_img(name="Ground Truth label train pancreas" + str(count_pancreas),im_slice = slice_y,writer=self.file_writer)
                    if count_pancreas == num_images and count_background == num_images:
                        break
                else:
                    if count_background!=num_images:
                        count_background+=1
                        z_slice = np.random.randint(0,x.shape[-1])
                        slice_x = x[:,:,:,z_slice]
                        self.img_patches_train_background.append((x,z_slice))
                        paint_img(name="Ground Truth image train background" + str(count_background),im_slice = slice_x,writer=self.file_writer)
                        if count_pancreas == num_images and count_background == num_images:
                            break
            if count_pancreas == num_images and count_background == num_images:
                break
        
        count_pancreas = 0
        count_background = 0
        self.img_patches_val_pancreas=[]
        self.img_patches_val_background=[]
        for batch in val_dataset:
            for i in range(batch[0].shape[0]):
                x = batch[0][i,:,:,:,:]
                y = batch[1][i,:,:,:,:]
                y = np.argmax(y,axis=0)
                if has_labels(y) and count_pancreas!=num_images:
                    count_pancreas+=1
                    slice_y, z_slice = get_slice(y)
                    slice_y = np.expand_dims(slice_y,axis=0)
                    slice_x = x[:,:,:,z_slice]
                    self.img_patches_val_pancreas.append((x,z_slice))
                    paint_img(name="Ground Truth image validation pancreas" + str(count_pancreas),im_slice = slice_x,writer=self.file_writer)
                    paint_img(name="Ground Truth label validation pancreas" + str(count_pancreas),im_slice = slice_y,writer=self.file_writer)
                    if count_pancreas == num_images and count_background == num_images:
                        break
                else:
                    if count_background!=num_images:
                        count_background+=1
                        z_slice = np.random.randint(0,x.shape[-1])
                        slice_x = x[:,:,:,z_slice]
                        self.img_patches_val_background.append((x,z_slice))
                        paint_img(name="Ground Truth image validation background" + str(count_background),im_slice = slice_x,writer=self.file_writer)
                        if count_pancreas == num_images and count_background == num_images:
                            break
            if count_pancreas == num_images and count_background == num_images:
                break

    
    def on_epoch_end(self, epoch, logs=None):
        
        count=0
        for (img, z_slice) in self.img_patches_train_pancreas:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction train pancreas " + str(count),im_slice=res_slice,writer=self.file_writer)

        
        count=0
        for (img, z_slice) in self.img_patches_train_background:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction train background " + str(count),im_slice=res_slice,writer=self.file_writer)


        count=0
        for (img, z_slice) in self.img_patches_val_pancreas:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction validation pancreas " + str(count),im_slice=res_slice,writer=self.file_writer)


        count=0
        for (img, z_slice) in self.img_patches_val_background:
            count+=1
            img = np.expand_dims(img,axis=0)
            res_model = self.model.predict(img)
            res_model = res_model[0,:,:,:,:]
            res = np.argmax(res_model,axis=0)
            res_slice = res[:,:,z_slice]
            res_slice = np.expand_dims(res_slice,axis=0)
            paint_img(name="Prediction validation background " + str(count),im_slice=res_slice,writer=self.file_writer)
