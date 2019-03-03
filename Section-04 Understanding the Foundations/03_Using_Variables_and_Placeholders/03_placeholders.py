
# coding: utf-8

# # Placeholders
# 
# We introduce how to use placeholders in TensorFlow.
# 
# First we import the necessary libraries and reset the graph session.

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Start graph session

# In[7]:


sess = tf.Session()


# ### Declare a placeholder
# 
# We declare a placeholder by using TensorFlow's function, `tf.placeholder()`, which accepts a data-type argument (`tf.float32`) and a shape argument, `(4,4)`.  Note that the shape can be a tuple or a list, `[4,4]`.

# In[8]:


x = tf.placeholder(tf.float32, shape=(4, 4))


# For illustration on how to use the placeholder, we create input data for it and an operation we can visualize on Tensorboard.
# 
# Note the useage of `feed_dict`, where we feed in the value of x into the computational graph.

# In[9]:


# Input data to placeholder, note that 'rand_array' and 'x' are the same shape.
rand_array = np.random.rand(4, 4)

# Create a Tensor to perform an operation (here, y will be equal to x, a 4x4 matrix)
y = tf.identity(x)

# Print the output, feeding the value of x into the computational graph
print(sess.run(y, feed_dict={x: rand_array}))


# To visualize this in Tensorboard, we merge summaries and write to a log file.

# In[10]:


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/7", sess.graph)


# We run the following command at the prompt:
# 
# `tensorboard --logdir=/tmp`
# 
# Which will tell us where to navigate chrome to to visualize the computational graph.  Default is
# 
# `http://0.0.0.0:6006/`

# ![Placeholders_in_Tensorboard](https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/images/03_placeholder.png)
