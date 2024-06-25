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


fig, ax = plt.subplots(figsize=(8, 8))
line = [x for x in range(1, 9)]
line1 = [1 for x in range(1, 9)]
line2 = [8 for x in range(1, 9)]
ax.axis((0, 9, 0, 9))
ax.plot(line1, line, color='r')
ax.plot(line2, line, color='r')
ax.plot(line, line2, color='r')
for goal in env.goalstates:
    ax.plot(goal[0], goal[1], 'ro', color='g')    
ax.grid(True)
bot, = ax.plot([],[],'ro', color='y')
ball, = ax.plot([],[],'ro', color='b')
reward = ax.set_title('Reward=' + str(env.currentstate[1]) + ' End=' + str(env.currentstate[2]))

def update(frame,env,bot,ball,reward):
    global s,d    
    a = soccerpredictwiths(s)
    
    s,r,d = env.step(a)
    bot.set_data(env.currentstate[3], env.currentstate[4])
    ball.set_data(env.currentstate[5], env.currentstate[6])
    reward.set_text('Reward=' + str(env.currentstate[1]) + ' End=' + str(env.currentstate[2]))
    print(bot, ball, reward,a)
    return bot,ball,reward



ani = FuncAnimation(fig, update,frames=10,fargs=(env,bot,ball,reward))    
ani.save('./docs/test.gif', writer='imagemagick')
plt.show()    
    
 