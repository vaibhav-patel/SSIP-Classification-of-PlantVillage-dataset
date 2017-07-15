
# coding: utf-8

# In[10]:


from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
                                            


# In[11]:



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
model=loaded_model


# In[12]:


import os
ls1=os.listdir('color')
dic1={}
import scipy.misc as sm
import numpy as np
count=0
for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('color/'+i)
    for j in ls2:
        #im1=np.asarray(sm.imread('color/'+i+'/'+j))
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        count=count+1
        
    
dic2={}
for i in dic1:
    dic2[dic1[i]]=i


# In[ ]:





# In[16]:


def test_it():
    xvap=os.listdir('I_want_to_test_it/')
    for i in xvap:
        im1=np.asarray(sm.imread('I_want_to_test_it/'+i))
        im1=im1/255.0
        im1=im1.reshape((1,im1.shape[0],im1.shape[1],im1.shape[2]))
        print(model.predict(im1))
        xx=(np.argmax(model.predict(im1)))
        print (dic2[int(xx)])
        os.rename('I_want_to_test_it/'+i,'I_want_to_test_it/'+dic2[int(xx)]+i)
     


# In[18]:


test_it()


# In[ ]:




