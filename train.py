import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


rg=range(410000,415000)  # the size of the dataset 
path=r"data/"            # dataset path
load_mode=True           # to update the previous model     
start=time.time()

def load_images_from_folder(path,rg):
    images = []
    counter=0

    obs1=np.load(path+"obs.npy")
    obs=obs1[rg]
    for filename in obs:
        img = cv2.imread(path+filename,cv2.IMREAD_UNCHANGED)
        counter+=1
        if img is not None:
            #img=cv2.resize(img,(32,32))
            images.append(img)
    return images 


def create_train_test(rg):
    images =load_images_from_folder(path,rg)
    action1=np.load("data/actions.npy")
    action=action1[rg]
    train_images=np.array(images[0:tr_num])
    train_labels=action[:tr_num]
    test_images=np.array(images[tr_num:num])
    test_labels=action[tr_num:num] 
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images,train_labels,test_images,test_labels
    

num=len(rg)
tr_num=int(num*0.75)

train_images,train_labels,test_images,test_labels=create_train_test(rg)

inputshape=(train_images.shape[1],train_images.shape[2],train_images.shape[3])
R1=np.load("data/rewards.npy")
R2=R1[0:tr_num]
R=R2.reshape(tr_num)+1       # can be used with Fit
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=inputshape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4))
model.summary()

if load_mode==False:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('MyModel_tf410k')

early_stopping_monitor = EarlyStopping(patience=1)

#sample_weight=R, 
history = model.fit(train_images, train_labels,sample_weight=R,    batch_size=32, epochs=10, 
                    validation_data=(test_images, test_labels),callbacks=[early_stopping_monitor])

# saving the model in tensorflow format
model.save('Model_tf',save_format='tf')



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

prd=model.predict(np.array([test_images[0]])).argmax()
print("prediction of an image [0] is ", prd)
print("Time=", time.time()-start)
