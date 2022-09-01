import gym
import matplotlib.pyplot as plt
import cv2
from collections import deque
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time


start=time.time()
def gray_sized(obs):
    obs_gray= cv2.cvtColor( obs, cv2.COLOR_RGB2GRAY)
    obs_gray_sized =cv2.resize(obs_gray,(84,84),interpolation=cv2.INTER_AREA)
    return obs_gray_sized



def process_obs(queue):
    x1=np.max(np.array([queue.popleft(),queue.popleft()]),axis=0)
    x2=np.max(np.array([queue.popleft(),queue.popleft()]),axis=0)
    x3=np.max(np.array([queue.popleft(),queue.popleft()]),axis=0)
    x4=np.max(np.array([queue.popleft(),queue.popleft()]),axis=0)
    return np.stack([x1,x2,x3,x4],axis=-1)/255.0

model = tf.keras.models.load_model('MyModel_tf410k')
env=gym.make("Breakout-v0")
scores = []
choices = []
count=0
for episode in range(5):
    score = 0
    prev_obs = []
    queue=deque([])
    obs=env.reset()
    for i in range(8):
        queue.append(gray_sized(obs))

    obs_final=process_obs(queue.copy())
    
    done=False
    while not done:
        
        if count%20==0:
            action =1   #to break the frozen cases
        else:
            action=model.predict(np.array([obs_final])).argmax() 

        obs, reward, done, info = env.step(action)
        queue.popleft()
        queue.append(gray_sized(obs))
        obs_final=process_obs(queue.copy())

        score+=reward
        env.render()
        count+=1
    scores.append(score)
    print()
    print("Episode {} Score {}".format(episode+1,score))
    

env.close()

print()
print('Average Score:', sum(scores)/len(scores))
print('max score:', max(scores))
print("Time=", time.time()-start)
print("Iterations=",count)