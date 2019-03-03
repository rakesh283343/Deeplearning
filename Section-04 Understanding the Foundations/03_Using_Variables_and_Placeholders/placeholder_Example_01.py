import tensorflow as tf
# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf . placeholder ( tf . float32 , shape =[ 3 ])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf . constant ([ 5 , 5 , 5 ], tf . float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
#If we try to fetch c , we will run into error.
with tf . Session () as sess:
    print ( sess . run ( c, { a : [ 1 , 2 , 3 ]} ))



# create Operations, Tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)
# start up a `Session` using the default graph
sess = tf.Session()
# define a dictionary that says to replace the value of `a` with 15
replace_dict = {a: 15}
# Run the session, passing in `replace_dict` as the value to `feed_dict`

print(sess.run(b, feed_dict=replace_dict)) # returns 45
print(sess.run(b, { a : (5)})) # returns 45

writer = tf.summary.FileWriter("/tmp/8", sess.graph)