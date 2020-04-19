# <p align="center"> Pancreas tumor image segmentation </p>

### <p align="center"> UPC Artificial intelligence with Deep Learning Postgraduate Course </p>

##### <p align="center"> Students: Roger Borràs, Jaume Brossa, Cristina De La Torre, Cristian Pachón </p>

##### <p align="center"> Advisor: Santi Puch </p>

### Repository structure

Here is a brief description on what can be found in the different files and directories of this repository:

**images:** image files used in this README

**notebooks:** different files (scripts and notebooks) used for testing while developing the final scripts used for the project

**utils:** directory containing the different support python scripts used by the main ones:

* **callback_functions.</span>py:** homemade callback functions for showing sample images on tensorboard while training.

* **patches.</span>py:** functions used for separaing images into patches.

* **transformation.</span>py:** data augmentation functions.

**data.</span>py:** data management functions. 

**homemade_unet.</span>py:** homemade implementation of the 3D U-Net used in this work.

**metrics.</span>py:** Implementation of the different metrics used in this project.

**train.</span>py:** Training loop.


### Motivation
All of us are interested in biomedical applications and a central problem in health is the patient diagnostic process. This figure represents the general pipeline for patient diagnosis in health care:

<p align="center">
    <img align="center" src="images/motivation_data/dp.png">
</p>

* A first step, the engagement where a patient experiences a health problem
* Second, define the problem with medical data collection and interpretation
* And finally, treatment and evaluate the outcomes

For the data collection in patient diagnosis, there are different data sources like clinical examination, health monitoring, laboratory results and medical imaging. 

<p align="center">
    <img align="center" src="images/motivation_data/md.png">
</p>

Medical imaging is a very important tool in diagnosis, and it will be more important in the future without any doubt. Our project is focused on apply Deep Learning to this kind of data to improve diagnosis. In a recent paper from New England journal of Medicine (NEJM), they suggest that neural networks has the potential to revolutionize health in the future.

<p align="center">
    <img align="center" src="images/motivation_data/NEJM_1.png">
</p>

### About Medical Segmentation Datathon
To practice this type of problems, we searched for data from Medical Segmentation Dechatlon Competition and we picked the pancreas tumor segmentation problem:

<p align="center">
    <img align="center" src="images/motivation_data/MSD.png">
</p>

In this [link](http://medicaldecathlon.com/), there are all the details about panceas competition as well as other medical competitions.


### Input dataset
The input dataset for this task consists in **282 3D volume images** (for training and validation) of size **512 x 512 x number of slices**, each 3D image is composed by a specific number of slices (around 100) of size 512x512. The **modality** used to take these images was **computed tomography (CT)**. This modality takes images of the body from several angles in order to compose the final 3D image of this part scanned, in our case a volume of the thorax, place where pancreas could be found. 

These images have been acquired from the **Memorial Sloan Kettering Cancer Center** (MSK), a highly recognized institute focused on cancer research.

<p align="center">
<img  src="images/input_dataset/ct__scan.gif?raw=true" alt="CT scanner" title="CT scanner"  width="400"> 
</p>

Normally in datasets like ImageNet, images have the *jpg* format or similar. However in this medical task,  the ***NifTI*** format is used. The NifTI format is commonly used in medical imaging, this format not only provides the image information (numpy array of the image) but also a header with metadata, where some relevant points were given. These relevant points consist mainly on the properties of the equipment where these images have been taken, data about the patient, etc. This information has been fundamental during years to radiologist in order to segment and see the images properly, so it will be also relevant for us to deal with this task successfully.

### Purpose of our project
Our purpose consists in **segment automatically** from a raw CT image of the thorax, the **pancreas** and its **tumor**. One example of **what we had** and **what we wanted to achieve** is presented in the following image. As we can see below, we had a raw image difficult to visualize at first sight and its label.

<p align="center">
<img  src="https://i.imgur.com/fHGWnbJ.png" alt="Raw image and its respective label" title="Raw image and its respective label"  width="500" > 
</p>

At first, these images have been preprocessed applying some special techniques normally used in CT medical images for machine and deep learning problems. Then we customized an input pipeline for these types of 3D medical images to finally pass them to a customized network, also prepared for 3D input images.

The result that we wanted to obtain, consisted of a 3D image segmented. This output, as you we see in the above image, have three different colors. Each color correspond to one structure presented on the raw image:

* Black, background
* White, pancreas
* Grey, pancreas tumor

So what we wanted during this project was to classify each pixel of a 3D input image into this three different classes (background, pancreas and its tumor) and finally obtain a 3D output image with the segmented pancreas and tumor.

### Challenges faced
As mention before, the input dataset consisted of 3D images. It meant that **a high amout of time** was needed to train the model. Since the architecture of the network was quite deep (as it is explained in *Architecture* section), it took a lot of time **to make the model learn**. In this section, there are described the actions done in order to solve these problems.

#### High amout of time
It was observed the model needed a large amount of time to train. The first hyphotesis was the images were too big. In order to validate this hypotheis, we did the following changes:

* Resizing: It actually lowered the time needed but at the same time, a lot of information was lost. So, we decided not to use it

<p align="center">
    <img align="center" src="images/challenges_faced/resize.png">
</p>

* Crop: Image crop along axes x, y and z. Relevant information was kept and input was smaller. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/crop.png">
</p>

* Patches: Division of the the image into patches. It helped to solve computational problems. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/patches.gif">
</p>

#### Deal with unbalanced data
 After analysing the dataset, we saw the data was highly unbalanced. The following figures shows the raw image and the pancreas tumor:

<p align="center">
    <img align="center" src="images/challenges_faced/raw_image.png">
</p>

<p align="center">
    <img align="center" src="images/challenges_faced/target.png">
</p>

As we can see, pancreas tumor is a really small portion of the image. 

We wanted to avoid the model to assign all the pixels to the backgroud class. For that reason we developed two solutions:

* Use a problem-oriented loss function. A lot of similar-task papers we studied mentioned the **Generalised Dice Loss** (GDL) was the indicated loss to use for this problem.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=1 - 2 \frac{\sum_{}^{2}w_l\sum_{n}^{}r_{ln}p_{ln}}{\sum_{}^{2}w_l\sum_{n}^{}\big(r_{ln} + %2B p_{ln}\big)}">

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=w_l = \frac{1}{ \big(\sum_{n}^{}r_{ln} \big)^2 }">
</p>

* Balance the data in order the have the same amount of pancreas class as background class. We did that by undersampling background patches and applying data augmentation to pancreas/tumor patches.

### Architecture
In this section, it is described the architecture of our proposal. This architecture is composed by two main parts:

* **Input pipeline**:  process where images are modified and prepared to enter on our network properly 
* **Network**: process where our model learns and predicts the pancreas and its tumor segmentation

#### Input pipeline
As mentioned earlier, our input images consists of 3D CT medical images so we had to take this into account in order to manage them suitably. Before starting with the input pipeline, every CT medical image should be subjected to some **preprocessing**. This preprocessing is performed in addition to some other transformations in the **input pipeline** designed.

During preprocessing, various techniques were carried out in order to obtain an **improved image** and ready to enter on the model. The **main changes applied to these images** are the following:

* **Pixel spacing resampling**:
	 An special point taken into account was the pixel spacing in this 3D images. As we can see in the image below, we have the sizes of the pixels and the size of the voxel (distance between slices).  This pixel spacing depends on the scanner used, so in this way we could be dealing with images with different properties. It is important to keep the pixel spacing consistent or, on the contrary, it may be hard for our network to **generalize**. So, we wanted to have same sizes for all the images.

	We resampled all the images to a pixel spacing of type [1, 1, 1] . With this technique, a neural network will improve the predictions in different images since they will have the same spatial information and the kernels can learn this information in the same way.

	<p align="center">
	<img  src="https://i.imgur.com/RK9sk3a.png" alt="(A) CT image (B) Image array (C) Voxel with sizes axbxc" title="(A) CT image (B) Image array (C) Voxel with sizes axbxcl"  width="400" > 
	</p>

* **Windowing**:
		The intensities of grays in these types of medical images are not as usual as in normal images. Normal png files are usually in the standard range of 0-255 intensities values, while CT images can present  a high  range of intensities, from -1000 to 4000.
		
	This anormal range of intensities is due to the special unit of measurement in CT scans. This unit is the  **Hounsfield Unit**  **(HU)**  which is a measure of radiodensity. Each  **voxel**  (3D pixel) of a CT scan has an  **attenuation value**  that is the measure of the reduction intensity of a ray of light by the  **tissue**  through it passes. Each pixel is assigned to a numerical value called  **CT Number**, which is the average of all the attenuation values contained within the corresponding voxel. In the following image you will see this hounsfield scale, and some components or parts of the body are indicated, such as the pancreas.

	<p align="center">
	<img  src="https://i.imgur.com/XjwNd0j.jpg" alt="Hounsfield Units scale" title="Hounsfield Units scale"  width="800" > 
	</p>
	
	Since neither we nor the machines can recognize 4000 shades of gray easily, a technique called **windowing** has been used to limit the number of Hounsfield units that are displayed. For example if we want to examine the soft tissue in one CT scan we can use a **minimum pixel value** of -135 and a **maximum pixel value** of 250, the tissues with CT numbers outside this range will appear either black or white. A narrow range provides a higher contrast.

	In our case, pancreas was our main interest. After some research we found that this organ was difficult to segment alone by this HU because there were more organs with similar intensity. But we could eliminate the intensities that were not interesting for us like the bones or the air, so finally we decided to use the soft tissue window explained above.

	<p align="center">
	<img  src="https://i.imgur.com/PgE70L3.png" alt="(Right) Histogram before applying windowing (Left) Histogram after applying windowing" title="(Right) Histogram before applying windowing (Left) Histogram after applying windowing" width="700" >
	</p>

	So thanks to **HU**, radiologist have been able to differentiate all the tissues presented in a CT scanner during years, and also we have been able to improve and **facilitate the learning of our model**.

* **Histogram equalization**:
	Histogram equalization is a very popular technique used in order to improve the appearance and contrast of medical images. Histogram equalization is a technique where the histogram of the resultant image is as flat as possible (image below). This allows the areas of lower local contrast to gain a higher contrast. 
	
	<p align="center">	
	<img  src="https://i.imgur.com/zcEnXmX.png" alt="(Rigth) Part of the histogram before applying equalization (Left) Part of the histogram after applying equalization" title="(Right) Part of the histogram before applying equalization (Left) Part of the histogram after applying equalization" width="700" > 
	</p>

*  **Normalization**: 
	Pixel intensity values has been normalized between values 0 and 1. This normalization was applied to the whole image before splitting it into patches, to avoid different normalization parameters for different parts of the same image.
		
Once all these steps were applied, we got an improved image (as you can see in the below image), and we were able to start with the **input pipeline**. We have created two different input pipelines depending on we were dealing with images of training or validation dataset.

<p align="center">
<img  src="https://i.imgur.com/bbvS4DG.png" alt="(A) Raw image (B) Improved image" title="(A) Raw image (B) Improved image" width="500" > 
</p>

 ##### Training input pipeline 
<p align="center">
<img  src="https://i.imgur.com/tl74N1H.png" alt="Training input pipeline figure" title="Training input pipeline figure" >
</p>

This figure explains the flow of our training input pipeline. At first, we have the input image and its label, that are treated in the same way as numpy arrays. We divide these numpy arrays in different 3D volume patches of sizes 128x128x64. 

This patches can include or not the pancreas and its tumor, so as we have the label during training, we divide them into background patches and pancreas patches. This type of problem, as we have mentioned before is highly imbalance. So once we have the division, we apply **data augmentation** to the pancreas patches in order to increase the number of them until obtain 1.5 background patches for each pancreas patch.

<p align="center">
<img  src="https://i.imgur.com/sNDgHnh.png" alt="(Right) Image and label patch before transformations (Left) Image and label patch after transformations" title="(Right) Image and label patch before transformations (Left) Image and label patch after transformations" width = "300" >
</p>

Data augmentation has been made mild, in order to not apply too much noise to the model. Different techniques have been included such as flips, elastic deformations, add noise and offset. Specially, elastic deformation was recommended in a most of the papers we studied to increase the accuracy in this type of medical segmentation problems.

Once we have the background patches selected and the pancreas patches augmented, we concatenate and shuffle them.

 ##### Validation input pipeline
<p align="center">
<img  src="https://i.imgur.com/T1AF0Qz.png" alt="Validation input pipeline figure" title="Validation input pipeline figure" width="700">
</p>

This figure explains the flow of our validation input pipeline. At first, we have the input image (that was also treated as a numpy array). As in the training input pipeline we divide this numpy array in different 3D volume patches, but in this case the size is bigger, 256x256x64. 

#### Network
Our model was based in the **U-Net network**. The U-Net is a convolutional network architecture for fast and precise segmentation of images. We developed an implementation of this U-Net architecture from scratch using tensorflow/keras but by using 3D Convolutional layers to be able to work with 3D images. 

<p align="center">
<img  src="https://i.imgur.com/oDbdCuY.png" alt="U-Net for pancreas image segmentation" title="U-Net for pancreas image segmentation" width = "500" >
</p>

This network architecture consists of a contracting path (left side) and an expansive path (right side), which gives it the u-shaped architecture. The contracting side,  is a typical convolutional network, that consists of repeated application of convolutions, followed by Rectified linear unit (ReLU) and max poolings. While during the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path.

### Iterations
In order to make small but relevant improvements, several iterations have been done. In this section, they are described.

###### NOTE: all experiments where performed using Google Colab free GPU's and one single Nvidia Tesla K80 GPU in google cloud platform. 

##### Overfitting
The first step done was make the model overfit. To achieve it, the following configuration was used:

* Used pixel-wise Binary Cross Entropy Loss
* Used only 15 patches for training, 5 for validation
* Trained for around 370 epochs (9 hours)

The following figure shows the traning and validation loss:

<p align="center">
    <img align="center" src="images/iterations/loss_overfit.png">
</p>

We can see that from epoch 200 the training loss is still descending but the validation loss is flat. 

The following figures show how the model was overfitted. The first figure is a prediction for an image from the traning set and the second one is a prediction for an image from the validation set:

<p align="center">
    <img align="center" src="images/iterations/train_overfit.png">
</p>

<p align="center">
    <img align="center" src="images/iterations/validation_overfit.png">
</p>

The top left figure is an image from the training dataset (ground truth). The top right figure is its prediction. The bottom left figure is an image from the validation dataset(ground truth) and the bottom right image is its prediction.

Clearly, the model is memorising the training set and, thus, it is unable to predict anything for an unseen image, since everything is assigned to background class.

This experiment also shows the fact that cross entropy is not a good loss for our task, since it's returning good values for validation loss. This is because cross entropy doesn't take into account the unbalanced pixel problem we  described before and still gives a good score when the model is predicting everything to be background. 

#### Using Generalised Dice Loss
The next step was changing the loss function to a more appropriate one. Generalised Dice Loss has shown great success in similar tasks since it takes into account the number of pixels found in each label. With this new loss, another overfitting task was performed to demonstrate if our model has the ability to learn. The used settings were as follows:

* Used Generalized Dice Loss, which optimum value in our implementation is -1
* Used only 15 patches for training, 5 for validation
* Trained for around 230 epochs (5 hours)

The following figure shows the training and the validation loss function:

<p align="center">
    <img align="center" src="images/iterations/loss_dice.png">
</p>

The overfit is clear and we can see the validation loss working as expected. Let's check the predictions of the model:

<p align="center">
    <img align="center" src="images/iterations/train_dice.png">
</p>

<p align="center">
    <img align="center" src="images/iterations/validation_dice.png">
</p>

The top left figure is an image from the training dataset (ground truth). The top right figure is its prediction. The bottom left figure is an image from the validation dataset(ground truth) and the bottom right image is its prediction.

We can observe that the model has been able to start memorizing the training samples. Also, it is interesting to notice that the model is able to predict pancreas-like-shapes on the validation dataset even though overfitting is clear. This shows that the loss is appropriate for the task and that our model is capable of learning.

#### First trial with the whole dataset
After having all the ingredients ready, we performed an iteration using a subsample of 150 images (to reduce training time). The following configuration was used:

* Used Generalized Dice Loss
* Used 120 images for training, 30 for validation
* Used balanced data input + data augmentation

(Data augmentation is not strictly needed since we are using a subset of images, nevertheless we wanted to test if our data augmentation pipeline worked correctly and could be useful on more appropriate settings)

This configuration provided the following metrics:

<p align="center">
    <img align="center" src="images/iterations/loss_first.png">
</p>

Everything looked fine, but there was an **error**. We used the same input pipeline (balanced classes + data augmentation) for training and for validation. It was **fixed** using a normal input pipeline for validation without data augmentation: more realistic strategy for inference.

After fixing it, we were ready to perform the final iteration. It is described in *final results* section.

### Final results

The results showed in this section are by no ends finished since we haven't been able to train the model long enough to either get meaningful loss values or tune hyper-parameters. This is due to the lack of time and computational resourcess we had available for this project.

Nevertheless, we understand that they reach what was intended in the scope of the course and this final project.

The hyperparameters from our last training were the following:

|          Hyperparameter         |    Value   |
|:-------------------------------:|:----------:|
|           Architecture          |  3D U-Net  |
|      Initial learning-rate      |    0.001   |
|            Batch size           |      2     |
|              Epochs             |     78     |
|         Train patch size        | 128x128x64 |
|      Validation patch size      | 256x256x64 |
|        # images training        |     120    |
|       # images validation       |     30     |
| Background/Pancreas patch ratio |     1.5    |

The loss metrics resulting of the last training are the following:

<p align="center">
     <img align="center" src="images/final_results/final_losses.png">
</p>

And some model predictions:

<p align="center">
     <img align="center" src="images/final_results/final_samples.png">
</p>

We can observe that after 78 epochs, the model is starting to learn to localize the pancreas and also starting to get an idea of the correct shape.

Taking into account that most of the papers we read took about 1000 epochs to get good results, we think that the actual results with only 78 epochs of training are quite good.

### Future work

As future work for this project, we plan on working in two different tasks:

1. Keep training the actual model and do hyper-parameter fine tunning to be able to find the neural network architecture that gives the best results. This includes trying different techniques used in other works like first segmenting only the pancreas and then segment the tumor with a second network.

2. Optimize the data input pipeline and training schedule by using more tensorflow native functions.
