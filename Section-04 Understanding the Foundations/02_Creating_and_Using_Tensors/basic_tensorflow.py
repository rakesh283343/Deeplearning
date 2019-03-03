import tensorflow as tf
print("PACKAGES LOADED")

hello1 = tf.constant('Hello, TensorFlow!')
session = tf.Session()
print("OPEN SESSION")

print(hello1)
print(session.run(hello1))

def print_tf(x):
    print("TYPE IS\n %s" % (type(x)))
    print("VALUE IS\n %s" % (x))
hello2 = tf.constant("HELLO. IT'S ME. ")
print_tf(hello2)


"""
import numpy as np
import tensorflow as tf
print("PACKAGES LOADED")

sess = tf.Session()
print("OPEN SESSION")

def print_tf(x):
    print("TYPE IS\n %s" % (type(x)))
    print("VALUE IS\n %s" % (x))
hello = tf.constant("HELLO. IT'S ME. ")
print_tf(hello)

hello_out = sess.run(hello)
print_tf(hello_out)
"""

a = tf.constant(1.5)
b = tf.constant(2.5)
print_tf(a)
print_tf(b)

a_out = session.run(a)
b_out = session.run(b)
print_tf(a_out)
print_tf(b_out)


a_plus_b = tf.add(a, b)
print_tf(a_plus_b)

a_plus_b_out = session.run(a_plus_b)
print_tf(a_plus_b_out)

a_mul_b = tf.multiply(a, b)
print_tf(a_mul_b)
a_mul_b_out = session.run(a_mul_b)
print_tf(a_mul_b_out)


# # VARIABLES

# In[10]:


weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)


# In[11]:


weight_out = session.run(weight)
print_tf(weight_out)


# # WHY DOES THIS ERROR OCCURS?

# In[12]:


init = tf.initialize_all_variables()
session.run(init)
print ("INITIALIZING ALL VARIALBES")


# # ONCE, WE INITIALIZE VARIABLES

# In[13]:


weight_out = session.run(weight)
print_tf(weight_out)


# # PLACEHOLDERS

# In[15]:


x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)


# ## OPERATION WITH VARIABLES AND PLACEHOLDERS

# In[18]:


oper = tf.matmul(x, weight)
print_tf(oper)


# In[21]:


data = np.random.rand(1, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)


# In[22]:


data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)

