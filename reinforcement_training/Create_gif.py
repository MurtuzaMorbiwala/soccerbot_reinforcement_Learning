import tensorflow as tf
from soccerworld_gym_env import SoccerWorld as sc
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

env = sc.Soccerworld()
import numpy as np
observations = env.countstate
actions = 4
e = 0.2  

inputs1 = tf.placeholder(shape=[1,observations],dtype=tf.float32)


def soccerpredictwiths(s):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('reinforcement_training/savemodel/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('reinforcement_training/savemodel/'))
        graph = tf.get_default_graph()
        W = graph.get_tensor_by_name("W:0")
        Qout = tf.matmul(inputs1,W)
        predict = tf.argmax(Qout,1)
        a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(observations)[s:s+1]})
        return a[0]


s = env.reset()
d=False
name = 0

def update(frame):
    global s,d    
    a = soccerpredictwiths(s)
    s,r,d = env.step(a)
    fig,ax = env.render()
    return fig,ax

fig,ax = update(0)

ani = FuncAnimation(fig, update,frames=10)    

ani.save('soccerworld_animation.gif',writer='ffmpeg')

    
    
