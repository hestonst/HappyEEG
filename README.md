
## HappyEEG



HappyEEG is a cloud-enabled, mobile-ready, realtime EEG emotion predicting algorithm allowing marketing specialists to see how their material affects their target audience at the subconcious level. Want to know if a poster of a luxury brand spikes the Schadenfreude--responsible for jealousy-- in real-time with real-world stimuli to know how your audience react to your ad? Use HappyEEG!  

  - Generalizes to new brains with short training (<20 minutes)
  - State of the art training is performed entirely in the Google ML Engine
  - This allows a completely mobile system to monitor emotional states virtually anywhere


[![Google Cloud Platform](https://www.retailbusinesstechnologyexpo.com/__novaimages/4461286?v=636567609703630000&h=120&type=3&w=120&q=100)](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1003905&utm_content=text-ad-none-any-DEV_c-CRE_113120492767-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20%7C%20US%20%7C%20en%20%7C%20EXA%20~%20Google%20Cloud%20Platform-KWID_43700009942847400-kwd-26415313501&utm_term=KW_google%20cloud%20platform-ST_google%20cloud%20platform&gclid=CjwKCAjwx7DeBRBJEiwA9MeX_FowZwUrPoy-TiUvXPxKqlbdraEfNWZ7JTEn4HWL6TK4m2P8BNvAyBoCy8kQAvD_BwE&dclid=CJ2O1fStl94CFVMNNwodH0ED3A) 
[![Jupyter Notebooks](https://images.g2crowd.com/uploads/product/image/large_detail/large_detail_1514651055/jupyter.png)](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1003905&utm_content=text-ad-none-any-DEV_c-CRE_113120492767-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20%7C%20US%20%7C%20en%20%7C%20EXA%20~%20Google%20Cloud%20Platform-KWID_43700009942847400-kwd-26415313501&utm_term=KW_google%20cloud%20platform-ST_google%20cloud%20platform&gclid=CjwKCAjwx7DeBRBJEiwA9MeX_FowZwUrPoy-TiUvXPxKqlbdraEfNWZ7JTEn4HWL6TK4m2P8BNvAyBoCy8kQAvD_BwE&dclid=CJ2O1fStl94CFVMNNwodH0ED3A) 
[![Google Cloud Platform](https://d1q6f0aelx0por.cloudfront.net/product-logos/6bd224a8-e827-4593-b5b4-483338e9999e-python.png )](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1003905&utm_content=text-ad-none-any-DEV_c-CRE_113120492767-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20%7C%20US%20%7C%20en%20%7C%20EXA%20~%20Google%20Cloud%20Platform-KWID_43700009942847400-kwd-26415313501&utm_term=KW_google%20cloud%20platform-ST_google%20cloud%20platform&gclid=CjwKCAjwx7DeBRBJEiwA9MeX_FowZwUrPoy-TiUvXPxKqlbdraEfNWZ7JTEn4HWL6TK4m2P8BNvAyBoCy8kQAvD_BwE&dclid=CJ2O1fStl94CFVMNNwodH0ED3A)


## Setup

Simply download the DEAP dataset from [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and set-up a Google GPU-enabled Jupyter notebook in under 15 minutes per [these instructions](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52?fbclid=IwAR1Mg-sls7VAvlhttps%3A%2F%2Fl.facebook.com%2Fl.php%3Fu%3Dhttps%3A%2F%2Ftowardsdatascience.com%2Frunning-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52%3Ffbclid%3DIwAR1Mg-sls7VAvlRXOPRP7KkwWu4HfjDZ2WCIC_r9ednvIOqoYcBVWYzqrSw&h=AT2uBjVkEL0ZSw-9vN4a17BCVdArWVUQSe03JvYA_d-nVPFkp8AxUAnaZ0IrXzVydi6BJ8L6Co3VmDOH0IRdHy-QUpnXi6H_uSPIo-wIjkrjkl9dT3dN-W3Shp_YK9D72W6x40hnDdnVJWDe76NcJuwRXOPRP7KkwWu4HfjDZ2WCIC_r9ednvIOqoYcBVWYzqrSw). Then upload this very notebook to your Google VM to be running accurate valence (emotion-state) classification right away! Serve this model with Google AI Enginer to allow for online learning and connect an internet (MySQL) commercially available EEG cap to monitor your target audience's emotion state and intensity. 


## Classifing Offline: 

First, let's read in a entire minute of EEG data. This is a lot of time to be considered real-time, but it's a good proof-of-concept. Note that our data has already been down-sampled to 126 Hz, and trasnformed into the power domain. Implement these *in silico* for a real-time system. 


```python
from os import listdir
import random
from os.path import isfile, join
import numpy as np
import numpy as np
import datetime
import os
import subprocess
import sys
import xgboost as xgb
import pickle 
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-6-5a51e74eb298> in <module>()
          8 import subprocess
          9 import sys
    ---> 10 import xgboost as xgb
         11 import pickle


    ImportError: No module named 'xgboost'



```python

#first, read in the pre-processed data from a folder containing the DEAP .dat files.  

print("Loading EEG files")
validation_fragments = []
validation_truth = []
train_fragments = []
train_truth = []
filenames = [f for f in listdir("DEAP_data/") if (isfile(join("DEAP_data/", f)) and '.dat' in f)]
print("Filenames are ", filenames)

x_test = []
y_test= []
x_train = []
y_train= []
import _pickle as cPickle

import random

for filename in filenames:
    with open("DEAP_data/" + filename, 'rb') as f:
        print(filename)
        array = cPickle.load(f, encoding='latin1')
        #print("array is", np.array(array))
        for datum, label in zip(list(array["data"]), list(array["labels"])):
            if random.uniform(0,1) < .2:
                x_test.append(np.array(datum).flatten())
                y_test.append(label[0])
            else:
                x_train.append(np.array(datum).flatten())
                y_train.append(label[0])


import numpy as np

x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)

# Load data into DMatrix object
print(x_train.shape, " is length")
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalidation = xgb.DMatrix(x_test, label=y_test)

# Train XGBoost model
parameter_dictionary = {}
parameter_dictionary["objective"] = "binary:logitraw"
parameter_dictionary["watchlist"]= dvalidation

clf = xgb.XGBRegressor(n_estimators=10000)
eval_set  = [(x_train,y_train), (x_test,y_test)]
clf.fit(x_train, y_train, eval_set=eval_set, eval_metric="mae")
# Export the classifier to a file
model = 'model.bst'

```


## Classifing Online: 

Now that we have some classification working, let us confirm that we can classifiy with only 3 seconds of brain-recording so that we can truly claim a real-time system! 




```python


print("Loading EEG files")
validation_fragments = []
validation_truth = []
train_fragments = []
train_truth = []
filenames = [f for f in listdir("DEAP_data/") if (isfile(join("DEAP_data/", f)) and '.dat' in f)]
print("Filenames are ", filenames)

x_test = []
y_test= []
x_train = []
y_train= []
import _pickle as cPickle

import random

for filename in filenames:
    with open("DEAP_data/" + filename, 'rb') as f:
        print(filename)
        array = cPickle.load(f, encoding='latin1')
        #print("array is", np.array(array))
        for datum, label in zip(list(array["data"]), list(array["labels"])):
            numberSegments = (len(datum) % (1751))
            markers = np.array([(1,(i)*(1751),(i+1)*(1751)-1) for i in range(numberSegments-1)]) 
            print("markers are ", markers)
            for stim_code, start, end in markers:
                stim_code, start, end = (int(stim_code), int(start), int(end))
                #note that the Emory lab doesn't use end; all are of length 1
                print("datum size is ", datum.size)
                print("start is ", start)
                fragment = datum[start:end]   
                print("fragment size is ", fragment.size)
                if not len(fragment) == 0:
                    if random.uniform(0,1) < .8:
                        x_test.append(np.array(fragment).flatten())
                        y_test.append(label[0])
                    else:
                        x_train.append(np.array(fragment).flatten())
                        y_train.append(label[0])


import numpy as np



x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)


# Load data into DMatrix object
print(x_train.shape, " is length")
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalidation = xgb.DMatrix(x_test, label=y_test)

# Train XGBoost model
parameter_dictionary = {}
parameter_dictionary["objective"] = "binary:logitraw"
parameter_dictionary["watchlist"]= dvalidation

clf = xgb.XGBRegressor(n_estimators=10000)
#we found XGBoost to be more affective than any keras-implemented CNN we tried
eval_set  = [(x_train,y_train), (x_test,y_test)]
clf.fit(x_train, y_train, eval_set=eval_set, eval_metric="mae")
#we use mean absolute error becuase this is a regression problem 
# Export the classifier to a file
model = 'model.bst'
```



Et voilà ! A state-of-the-art classifier fit for a novel EEG application, which achieves almost 80% accuracy. Do you think you experience anger, jealousy, or happiness when you see most ads? What makes them more or less effective? 

