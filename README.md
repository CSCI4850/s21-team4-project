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









    
    
    
    