
# coding: utf-8

# In[3]:

import tensorflow as tf
import ScoccerWorld as sc
env = sc.Soccerworld()
import numpy as np
observations = env.countstate
actions = 4
e = 0.1

inputs1 = tf.placeholder(shape=[1,observations],dtype=tf.float32)



def soccerpredict(xa,ya,xb,yb):
    s = env.getstate(xa,ya,xb,yb)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('C:/Users/mmorbiwala/Desktop/HackathonProject/PratikUpdatedFiles/Savemodel/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('C:/Users/mmorbiwala/Desktop/HackathonProject/PratikUpdatedFiles/Savemodel/'))
        graph = tf.get_default_graph()
        W = graph.get_tensor_by_name("W:0")
        Qout = tf.matmul(inputs1,W)
        predict = tf.argmax(Qout,1)
        a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(observations)[s:s+1]})
        return a[0]


# In[ ]:



