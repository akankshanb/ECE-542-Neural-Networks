#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from struct import unpack
from sklearn import metrics


# In[2]:


def load_mnist(img_path, label_path):
    images = open(img_path, 'rb')
    labels = open(label_path, 'rb')
    images.read(4)

    # Get metadata for images

    num_images = images.read(4)
    number_images = unpack('>I', num_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels

    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data

    x = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

train_img, train_lbl = load_mnist('train-images.idx3-ubyte'
                                 , 'train-labels.idx1-ubyte')
test_img, test_lbl = load_mnist('t10k-images.idx3-ubyte'
                               , 't10k-labels.idx1-ubyte')


# In[3]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_img, train_lbl, test_size=.25, random_state=0)


# In[4]:


x_train = x_train/255.0
x_val = x_val/255.0


# In[5]:


from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()


# In[6]:


lg.fit(x_train,y_train)


# In[7]:


score_val_lg = lg.score(x_val, y_val)
print(score_val_lg)


# In[8]:


from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(n_jobs=1, n_estimators = 10, max_features = 'auto',criterion = 'entropy')


# In[9]:


rf1.fit(x_train, y_train)


# In[10]:


score_val_rf1 = rf1.score(x_val, y_val)
print(score_val_rf1)


# In[11]:


rf2 = RandomForestClassifier(n_jobs=1, n_estimators = 20, max_features = 'auto',criterion = 'entropy')


# In[12]:


rf2.fit(x_train, y_train)


# In[13]:


score_val_rf2 = rf2.score(x_val, y_val)
print(score_val_rf2)


# In[14]:


train_img = train_img/255.0
test_img = test_img/255.0


# In[15]:


lg.fit(train_img, train_lbl)


# In[16]:


score_test_lg = lg.score(test_img, test_lbl)
print(score_test_lg)


# In[17]:


lg_predictions = lg.predict(test_img)


# In[41]:


rf2.fit(train_img, train_lbl)


# In[42]:


score_test_rf = rf2.score(test_img,test_lbl)
print(score_test_rf)


# In[43]:


rf_predictions = rf2.predict(test_img)


# In[44]:


lr_matrix = np.full((10000, 10), 0)
for index in range(0,11):
    lr_matrix[index][lg_predictions[index]] = 1


# In[45]:


rf_matrix = np.full((10000, 10), 0)
for index in range(0,11):
    rf_matrix[index][rf_predictions[index]] = 1


# In[46]:
lr_csv = np.asarray(lr_matrix)
np.savetxt("lr.csv", lr_csv.astype(int), fmt='%i', delimiter=",")

# In[ ]:

rf_csv = np.asarray(rf_matrix)
np.savetxt("rf.csv", rf_csv.astype(int), fmt='%i', delimiter=",")


