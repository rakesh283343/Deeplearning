import tensorflow as tf
import numpy as np
a = tf . constant ( 2)
b = tf . constant ( 3)
x = tf . add ( a , b)
# constant of 1d tensor (vector)
c = tf . constant ([ 2 , 2 ], name = "vector")
# constant of 2x2 tensor (matrix)
d = tf . constant ([[ 0 , 1 ], [ 2 , 3 ]], name = "b")
e = tf . zeros ([ 2 , 3 ], tf . int32 )
f = tf . zeros_like ([d] )
g = tf . ones ([ 2 , 3 ], tf . int32 )
h = tf . fill ([ 2 , 3 ], 8 )
i = tf . linspace ( 2.0 , 20.0 , 10 , name = "linspace" )
j = tf . range ( 3 , 33 , 3 )
k = tf . ones ([ 2 , 2 ],tf.float32 )
l = tf . ones ([ 2 , 2 ], np . float32 )
with tf . Session () as sess:
    writer = tf . summary . FileWriter ( './graphs' , sess . graph)
    print(sess . run (x))
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(f))
    print(sess.run(g))
    print(sess.run(h))
    print(sess.run(i))
    print(sess.run(j))
    print(sess.run(k))
    print(sess.run(l))
# close the writer when you’re done using it
writer.close()