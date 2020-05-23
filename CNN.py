
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


model.summary()


# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:





# In[12]:


model.summary()


# In[13]:


model.add(Flatten())


# In[14]:


model.summary()


# In[15]:


model.add(Dense(units=128, activation='relu'))


# In[16]:


model.summary()


# In[17]:


model.add(Dense(units=1, activation='sigmoid'))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:





# In[21]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
history=model.fit(
        training_set,
        steps_per_epoch=20,
        epochs=3,
        validation_data=test_set,
        validation_steps=800)


# In[ ]:





# In[22]:


model.save('mymodel.h5')


# In[23]:


#from keras.models import load_model


# In[24]:


#m = load_model('cnn-cat-dog-model.h5')


# In[25]:


#from keras.preprocessing import image


# In[26]:





# In[27]:


#type(test_image)


# In[28]:


#test_image


# In[29]:


#test_image = image.img_to_array(test_image)


# In[30]:


#type(test_image)


# In[31]:


#test_image.shape


# In[32]:


import numpy as np 


# In[33]:


#test_image = np.expand_dims(test_image, axis=0)


# In[34]:


#test_image.shape


# In[35]:


#history


# In[36]:


#result = model.predict(test_image)


# In[ ]:





# In[37]:


"""if result[0][0] == 1.0:
    print('dog')
else:
    print('cat')"""




accuracy=model.evaluate_generator(test_set)





with open('acc_file.txt','w') as f:
    f.write(str(accuracy[1]))












#hello #test
#finalteest
#test2
