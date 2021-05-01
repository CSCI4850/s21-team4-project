# s21-team4-project
## Neocognitron group 4
### Members: Zernab Saeed, Alejandro Lopez, Daniel Sindel, Pratap Karki, Michael Kwarteng


#### 0.1 USING CONVOLUTION NEURAL NETWORKS IN IMAGE CAPTIONING


Caption generation is a challenging artificial intelligence problem of generation human-readable
textual description given a photographic image. It requires both understanding from the domain of
computer vision and a language model from the field oof natural language processing. The objective
of this demo is help be able to use jupyter notebook to build a deep learning Neural Network that
generates a caption after analyzing an image. We will use the Encoder-Decoder Architecture to
build the network. To feed the model, we prepare data from the ImageNet data collection. The
Xception,which is a pre-trained Convolutional Neural Network that extracts picture features from
our dataset,serves as. the encoder’s output layer. We concatenated this model with a Long-Short
Term Memory layer that acts as a decoder which will decode the vector representation into the
corresponding output sequence using another recurrent hidden layer.
<br />
<br />

#### Installation
In order to be able to run ImageCaptionGenerator.ipynb file you need to have a Anaconda or JupyterHub on Biosim  in you computer.


- Install Anaconda 
    -[link to download Anaconda](https://www.anaconda.com/products/individual)
    -choose you os and download 64-Bit .exe file
    -once download is complete, open this file and follow all process to install 
    -After completing installation process, open Anaconda Navigator
    -You will see some pre-install aplication tools including Jupyter Notebook, JupyterLab, etc.
    -choose Jupyter Notebook and lunch it
    -You will see Notebook is opened in your default browser
- Open this link for JupyterHub on BioSim (For mtsu CS student with CS account holder only)
    - [Open this link and enter your MT CS id and password] https://jupyterhub.cs.mtsu.edu/biosim/user
- Create your Github account
- search out repository using search box located on top-left corner of your homepage by copying following repository name
    - CSCI4850/s21-team4-project
- Open our git repository  and copy ssh link from the 'Code' button located at right corner of the repository
- Open notebook terminal
- Clone the repository by using following command
     - git clone < repository  link>
- open recently cloned repository  and double click on 'ImageCaptionGenerator.ipynb' file. 
- You're now able to run the code

We begin by importing some of the tools we need to process our data. These tools will used to build/train
our neural network and matplotlib for plotting and visualization, and numpy for everything
<br />
<br />
<!-- Code Blocks-->
```python
    import tensorflow.keras as keras
    from tensorflow.keras import backend as K
    import numpy as np
    from IPython.display import display
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    %matplotlib inline
```
<br />

 First a few tools for this -particular- example...
 These tool are needed for this specific type of building/training
```python
    
    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.xception import preprocess_input,decode_predictions
```

```python
    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.xception import preprocess_input,decode_predictions
```
<br />

1.The Data
We used some of the common image dataset. We will use the encoder/decoder in processing the
building/training the data Below is how we loaded our data containing our images, make sure you
are in the right directory to be able to load the images. Remember these are grey-scale images.
<br />

Image folder used from previous work. We used the folder of images from Open Lab 6 provided by Dr. Philips
-[ola 6](https://www.cs.mtsu.edu/~jphillips/courses/CSCI4850-5850/private/Open_Lab_6.pdf)

image path and set the size of the images to 299x299
```python
    def grab_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
X = np.concatenate(
[grab_image('xception_example/images/image_%d.JPEG'%(i))
for i in range(100)])
X.shape
```
<br />
 Following output we will get: (100, 299, 299, 3)
 Above output is the dimension of the images we've set. 
 <br />
 <br />
 
Take note this is a 4D tensor of input data for the neural network. The first dimension is 100 since
there are 100 images, each of has the same pixels thus 299 x 299 pixels with a 2 x 3 dimension and
the last dimension of size 3 dor the three color channels of the image(red, green, blue). After the
above code is run, each image has already been preprocessed for input into the pre-trained network
that we will be using Goodgles’s Xception net.

Let's load up the model.This will load up the architecture and then the pre-trained weights from the internet...
<br />
<br />
```python
    model = Xception(weights='imagenet')
    #Just the first image that we say above...
    preds = model.predict(X[:,:,:,:])
    #Decode the results into a list of tuples (class, description, probability)
    #(one such list for each sample in the batch)
    for i in range (0,100):
        print("\nImage", i + 1)
        plt.imshow(image.array_to_img(X[i,:,:,:]))
        plt.show()
        W=(decode_predictions(preds,top=10)[i])
        Sum=0
        Sum+=((W[0])[-1])
        j=1
        print((W[0])[1], end='')
        while ((Sum<.45)&(j<5)): # Captions derived from the top 45% of probabilities within the top 5 predictions
            if ((W[j])[-1]>.04): # 4% is the minimum probability to qualify as a caption
                Sum+=((W[j])[-1])
            j=j+1
        print('\n')
```










    
    
    
    








    
    
    
    