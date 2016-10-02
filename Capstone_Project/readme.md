# Description
This a classical Computer Vision classification problem where we attempt to distinguish the images of cats and dogs in a given
dataset. The dataset is actually provided by Kaggle on their platform and can be downloaded from [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

Though, the whole dataset has not been used as it would have required a GPU for speeding up the training process. Instead a small
subset of the data is taken and is used to validate the accuracy of the model. It is highly recommended to run the model on a GPU 
for achieveing benchmark accuracy.

# Software Requirements
In order to successfully run the project the following software and packages are required:
* [Anaconda](https://www.continuum.io/downloads)
* [Anaconda Navigator](https://docs.continuum.io/anaconda/navigator)
* Python2.7 or Python3.4
* Numpy
* Scipy
* OpenCV
* [Keras](https://keras.io/#installation)
* TensorFlow or Theano as a backend for Keras
* Matplotlib
* PIL
* [h5py](http://docs.h5py.org/en/latest/build.html#install)

# Software Installation
Create an environment using anaconda navigator and python2.7. Install numpy, scipy, PIL, h5py, matplotlib, theano, TensorFlow and OpenCV from the search
package list. More details can be found [here](https://docs.continuum.io/anaconda/navigator-using).

Install keras using pip `sudo pip install keras`

* If you are using theano as backend, you may need to install a bunch of dependecies along with BLAS. Refer [this](http://deeplearning.net/software/theano/install.html)
* If you are on Windows, you will be required to install OpenCV using unofficial binaries from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
 
