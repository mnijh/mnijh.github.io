---
layout: page2
title: Fortran-Keras Bridge (FKB)
---
{% include toc %}

The Fortran-Keras Bridge (FKB) Package ([https://github.com/scientific-computing/FKB](https://github.com/scientific-computing/FKB)) offers the possibility to link your Fortran code to the opportunities and possibilities of the Keras/Tensorflow framework. In the following I show swiftly how to deploy FKB in a simple fortran project.

I assume that you already have some experience with neural networks and the Keras/Tensorflow network. And that you are familiar with developping Fortran code.


# Setup
---

First, let us clone the github repository of the FKB project. 
Navigate on your machine to your preferential working directory and clone the repo using
```bash
git clone https://github.com/scientific-computing/FKB
```

Then change into the created directory and build the Fortran modules using
```bash
cd FKB
bash build_steps.sh
```
Note that you need a working fortran compiler to do so. If you don't use gfortran change the compiler in the `build_steps.sh` file.

Next we download `Keras`, `Tensorflow` and `sklearn` which will be used for our tutorial. There are a multitude of options on how to do so, depending on your operating system, GPU vs CPU setup etc. We illustrate here a quick and dirty way to install the basic CPU versions using `pip`. For a more tailored guide please look here (Linux), here (OSX) or here (Windows).

Otherwise just type
```bash
# (you might want to use pip3 instead of pip)
pip install tensorflow
pip install keras 
pip install sklearn
```
That is it, let us begin with the magic.

# Train Model in Keras
---

Let's start with the probably most common use case. We train a network in Keras and implement it into a Fortran code. A jupyter notbook of all the steps shown here can be found at the bottom of the tutorial along with the other source files.

If you already have a trained Keras model you can jump to section ??. Note that not all features of Keras are supported within FKB. A list can be found here ?.

## Define Keras Model
In python we import all the necessary functions:

```python
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

Next we set up a small dictionary with all conifg parameters such that we don't need to rewrite all the functions if we change them. We will come back at their meeing when we use them.
```python
# Define Model Parameters
config = {'LR': 1.e-2,                 # Learning rate
          'Loss':'mse',                # Loss function
          'DimInput':7,                # Input Dimension
          'NeuronsDense1':512,         # Number of Neurons 
          'ActivationDense1':'relu',   # Actiavation function
          'NDense1':4,                 # Number of identical dense networks
          'NeuronsDense2':3,           # NUmber of final neurons (Output Dimension)
          'ActivationDense2':'linear', # activation in final layer
          'NDense2':1,                 # Number of repeating networks
          'Epochs':1000,
          'BatchSize':1024,
         }
```

Now as the most important step let us creat the model. We combine four dense layers with the same amount of Neurons given by `config['NeuronsDense1']` and the activation `config['ActivationDense1']`. As you can see in the dictionary above we use 512 Neurons and the relu activation. While the last layer only has as many neurons as output parameters and uses linear activation.
```python
model = Sequential()
# Do not use InputLayer by Keras--Fortran bridge does not yet recognize this parameter
model.add(Dense(config['NeuronsDense1'],input_dim=config['DimInput'], activation=config['ActivationDense1']))   
model.add(Dense(config['NeuronsDense1'], activation=config['ActivationDense1']))   
model.add(Dense(config['NeuronsDense1'], activation=config['ActivationDense1']))   
model.add(Dense(config['NeuronsDense1'], activation=config['ActivationDense1']))   
model.add(Dense(config['NeuronsDense2'], activation=config['ActivationDense2']))
```

The same code can be written more general like:
```python
model = Sequential()
# Do not use Input Layer Keras--Fortran bridge does not yet recognize this parameter
for i in range(config['NDense1']):
    if (i==0):
        model.add(Dense(config['NeuronsDense1'],input_dim=config['DimInput'], activation=config['ActivationDense1']))   
    else:
        model.add(Dense(config['NeuronsDense1'], activation=config['ActivationDense1']))   
for i in range(config['NDense2']):
    model.add(Dense(config['NeuronsDense2'], activation=config['ActivationDense2']))
```

We can display the model defined model using
```python
model.summary()
```
Next we choose an optimizer and a learning rate. The learning rate is also defined in the config dictionary while for the optimizer we choose a simple gradient descent method.
```python
optimizer = SGD(config['LR'])
```
Finally we compile the model using 
```python
model.compile(loss=config['Loss'],optimizer=optimizer)
```

## Prepare Data


## Train Model

# Move from Keras to Fortran
---

In order to translate the now trained Keras model we use the `convert_weights.py` function of FKB, located in the KerasWeightsProcessing directory of the FKB source. 
```bash
python3 convert_weights.py --weights_file keras_model.h5
```

This generated now the `keras_model.txt` file which can be processed by the neural fortran modules.

## Fortran project
Before we continue let us set up a small fortran project.
We write a simple module fot the numerical computation which can then be use in a fortran program.


### Module
```Fortran
MODULE my_dnn_test

INTEGER,PARAMETER :: nX = 7
INTEGER,PARAMETER :: nX = 4
CONTAINS

SUBROUTINE INIT()
    !! Load neural network
    IMPLICIT NONE

    call net % load('keras_model.txt')

    RETURN
END SUBROUTINE INIT

SUBROUTINE ESTIMATE_QUANTITY(X,Y)
    IMPLICIT NONE
    REAL(WP),DIMENSION(nX),INTENT(IN) :: X
    REAL(WP),DIMENSION(nY),INTENT(OUT) :: Y

    Y = net % output(X) 

    RETURN
END SUBROUTINE ESTIMATE_QUANTITY
END MODULE
```

### Main Programm
```Fortran
PROGRAMM dnn_test
    USE my_dnn_test
    IMPLICIT NONE
    
    REAL(WP),DIMENSION(nX) :: X
    REAL(WP),DIMENSION(nY) :: Y
    REAL(WP) :: T1,T2
    INTEGER :: i
    CHARACTER(LEN=1) :: stat
    !! Load Keras Model
    CALL INIT()

    WRITE(*,*)' Fortran-Keras Bridge Tutorial:'
    WRITE(*,*)'----------------------------------------------'
    DO
        WRITE(*,*)' Please provide an input for the neural network'
        DO i=1,nX
            READ(*,*)X(i)
        ENDDO
        WRITE(*,*)' Estimate Output'
        
        !! Measure time to call the neural network 
        CALL CPU_TIME(T1)
        !! Estimate: 
        CALL ESTIMATE_QUANTITY(X,Y)
        CALL CPU_TIME(T2)
        !! Print Output
        WRITE(*,*)'Y = ',Y
        WRITE(*,*)'Duration [s]: ',T2-T1
        WRITE(*,*)'----------------------------------------------'
        WRITE(*,*)' Exit [Y/n]'
        READ(*,*)stat
        IF (trim(stat).ne.'n') EXIT
    ENDDO

END PROGRAMM
```

### Makefile
```makefile
# Compiler
FC=gfortran 

# Neural Fortran Library 
LIBS=-lneural

default: my_dnn_test.o

my_dnn_test.o: my_dnn_test.f95
    $(FC) -o $@ -c $^

dnn_test: dnn_test.f95 my_dnn_test.o
    $(FC) $(LIBS) -c $@ $<
```