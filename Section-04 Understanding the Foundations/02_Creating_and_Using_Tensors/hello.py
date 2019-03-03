import tensorflow as tf
print("PACKAGES LOADED")


session = tf.Session()
print("OPEN SESSION")

hello1 = tf.constant('Hello, TensorFlow!')
print(hello1)
print(session.run(hello1))



def print_tf(x):
    print("TYPE IS\n %s" % (type(x)))
    print("VALUE IS\n %s" % (x))
hello2 = tf.constant("HELLO. IT'S ME. ")
print_tf(hello2)



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


weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)


# In[11]:
init = tf.global_variables_initializer()
session.run(init)
print ("INITIALIZING ALL VARIALBES")


weight_out = session.run(weight)
print_tf(weight_out)
