import tensorflow as tf
my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def())


#Declare variables

#create variable a with scalar value
a = tf . Variable ( 2 , name = "scalar" )
#create variable b as a vector
b = tf . Variable ([ 2 , 3 ], name = "vector" )



#create variable c as a 2x2 matrix
c = tf . Variable ([[ 0 , 1 ], [ 2 , 3 ]], name = "matrix" )
# create variable W as 784 x 10 tensor, filled with zeros
W = tf . Variable ( tf . zeros ([ 784 , 10 ]))

#The easiest way is initializing all variables at once using: tf.global_variables_initializer()

#init = tf . global_variables_initializer ()
#c = tf.add (a,b)

W1 = tf . Variable ( 10 )
#W1 . assign ( 100 )
assign_op = W1 . assign ( 100 )

with tf . Session () as sess:
    #init = tf.global_variables_initializer()
    #sess. run ( init)
    #print(sess.run(c))
    sess.run(W.initializer)
    print(sess.run(W))
    print(W.eval())
    #print(sess.run(W))

    sess.run(assign_op)
    print(W1.eval())  # >> 10




"""
#Note that you use tf.run() to run the initializer, not fetching any value.
#To initialize only a subset of variables, you use tf.variables_initializer() with a list of variables you want to initialize:

init_ab = tf . variables_initializer ([ a , b ], name = "init_ab")
with tf . Session () as sess:
    tf . run ( init_ab)

#You can also initialize each variable separately using tf.Variable.initializer



# create variable W as 784 x 10 tensor, filled with zeros
W = tf . Variable ( tf . zeros ([ 784 , 10 ]))
with tf . Session () as sess:
    tf . run ( W . initializer)

#Another way to initialize a variable is to restore it from a save file. We will talk about in a few weeks.
#Evaluate values of variables If we print the initialized variable, we only see the tensor object.

# W is a random 700 x 100 variable object
W = tf . Variable ( tf . truncated_normal ([ 700 , 10 ]))
with tf . Session () as sess:
    sess . run ( W . initializer)
    print(W)
#>> Tensor ( "Variable/read:0" , shape =( 700 , 10 ), dtype = float32)

#To get the value of a variable, we need to evaluate it using eval()
# W is a random 700 x 100 variable object
W = tf . Variable ( tf . truncated_normal ([ 700 , 10 ]))
with tf . Session () as sess:
    sess . run ( W . initializer)
    print(W . eval ())


#Assign values to variables We can assign a value to a variable using tf.Variable.assign()

W = tf . Variable ( 10 )
W . assign ( 100 )
with tf . Session () as sess:
    sess . run ( W . initializer)
    print(W . eval ()) # >> 10

#Why 10 and not 100? W.assign(100) doesn’t assign the value 100 to W, but instead create an
#assign op to do that. For this op to take effect, we have to run this op in session.

W = tf . Variable ( 10 )
assign_op = W . assign ( 100 )
with tf . Session () as sess:
    sess . run ( assign_op)
    print(W . eval ()) # >> 100


#Interesting example:
# create a variable whose original value is 2
a = tf . Variable ( 2 , name = "scalar" )
# assign a * 2 to a and call that op a_times_two
a_times_two = a . assign ( a * 2)
init = tf . global_variables_initializer ()
with tf . Session () as sess:
    sess . run ( init)
    # have to initialize a, because a_times_two op depends on the value of a
    sess . run ( a_times_two ) # >> 4
    sess . run ( a_times_two ) # >> 8
    sess . run ( a_times_two ) # >> 16

#TensorFlow assigns a*2 to a every time a_times_two is fetched.




W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(10))) # >> 20
    print(sess.run(W.assign_sub(2))) # >> 18

#Because TensorFlow sessions maintain values separately, each Session can have its own current value for a variable defined in a graph.

W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8
print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42
sess1.close()
sess2.close()

#You can, of course, declare a variable that depends on other variables.
#Suppose you want to declare U = W * 2
# W is a random 700 x 100 tensor

W = tf . Variable ( tf . truncated_normal ([ 700 , 10 ]))
U = tf . Variable ( W * 2)

#In this case, you should use initialized_value() to make sure that W is initialized before its value is used to initialize W.

U = tf . Variable ( W . intialized_value () * 2)


##InteractiveSession


sess = tf . InteractiveSession ()
a = tf . constant ( 5.0)
b = tf . constant ( 6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print ( c . eval ())
sess . close ()



## Placeholders and feed_dict

#To define a placeholder, we use:

# tf . placeholder ( dtype , shape = None , name = None)



# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf . placeholder ( tf . float32 , shape =[ 3 ])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf . constant ([ 5 , 5 , 5 ], tf . float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
#If we try to fetch c , we will run into error.
with tf . Session () as sess:
    print ( sess . run ( c ))

#>> NameError


#This runs into an error because to compute c, we need the value of a, but a is just a placeholder
#without actual value. We have to first feed actual value into a.

with tf . Session () as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    # fetch value of c
    print ( sess . run ( c , { a : [ 1 , 2 , 3 ]}))
#>> [ 6. 7. 8.]

# We can feed as any data points to the placeholder as we want by iterating through the data set and feed in the value one at a time.

with tf . Session () as sess:
    for a_value in list_of_a_values:
        print(( sess . run ( c , { a : a_value })))


#You can feed values to tensors that aren’t placeholders. Any tensors that are feedable can be fed. To check if a tensor is feedable or not, use:

#tf.Graph.is_feedable(tensor)
# create Operations, Tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.mul(a, 3)
# start up a `Session` using the default graph
sess = tf.Session()
# define a dictionary that says to replace the value of `a` with 15
replace_dict = {a: 15}
# Run the session, passing in `replace_dict` as the value to `feed_dict`
sess.run(b, feed_dict=replace_dict) # returns 45

#feed_dict can be extremely useful to test your model. When you have a large graph and just want to test out certain parts,
# you can provide dummy values so TensorFlow won’t waste time doing unnecessary computations.

"""



