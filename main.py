
# coding: utf-8

# In[1]:




from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
                                            


# In[2]:


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
        
    


# In[3]:


import os
ls1=os.listdir('color')
dic1={}
import scipy.misc as sm
import numpy as np
X=np.zeros((count,256,256,3))
Y=np.zeros((count,1))
vap=0

for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('color/'+i)
    for j in ls2:
        im1=np.asarray(sm.imread('color/'+i+'/'+j))
        X[vap,:,:,:]=im1
        Y[vap,0]=idx
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        vap=vap+1
        
    


# In[4]:


print (Y.shape)
np.random.permutation(5)


# In[5]:


print (dic1)


# In[6]:


batch_size = 10
num_classes = len(dic1)
epochs = 12

# input image dimensions
img_rows, img_cols = 256, 256


X /= 255.0



# In[7]:


ind=np.random.permutation(X.shape[0])
len_ind=ind.shape[0]
train_ind= ind[0: int(0.8*len_ind)]
val_ind= ind[ int(0.8*len_ind) : int(0.9*len_ind)]
test_ind= ind[ int(0.9*len_ind) : len_ind]
X=X[ind]
Y=Y[ind]



# In[8]:


sm.imsave('name.png',X[0])


# In[9]:


X_train=X[0:int(0.8*len_ind)]
X_val=X[int(0.8*len_ind):int(0.9*len_ind)]
X_test=X[int(0.8*len_ind) :len_ind]



# In[10]:


def visuals(num):
    sm.imsave('name'+str(num)+'.png',X_train[num])
    for i in dic1:
        if(dic1[i]== int(Y[num,0] ) ):
            print (i)
visuals(0)
visuals(1)
visuals(2)
visuals(3)


# In[11]:


Y_train=Y[0:int(0.8*len_ind)]
Y_val=Y[int(0.8*len_ind):int(0.9*len_ind)]
Y_test=Y[int(0.8*len_ind) :len_ind]



# In[12]:


# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)



# In[13]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[14]:


model.summary()


# In[15]:



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[16]:



model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))


# In[ ]:





# In[17]:


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[18]:


dic2={}
for i in dic1:
    dic2[dic1[i]]=i


# In[33]:


def test_it():
    xvap=os.listdir('I_want_to_test_it/')
    #print (xvap)
    for i in xvap:
        im1=np.asarray(sm.imread('I_want_to_test_it/'+i))
        im1=im1/255.0
        im1=im1.reshape((1,im1.shape[0],im1.shape[1],im1.shape[2]))
        print(model.predict(im1))
        xx=(np.argmax(model.predict(im1)))
        print (dic2[int(xx)])
        os.rename('I_want_to_test_it/'+i,'I_want_to_test_it/'+dic2[int(xx)]+i)
        
        
    


# In[34]:


test_it()


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON


# In[22]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[23]:



# later...


# In[24]:





# In[ ]:




