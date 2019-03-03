import numpy as np
import tensorflow as tf

IMG_SIZE = [3, 3]
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_VAL = 1

def get_training_batch(batch_size):
    ''' training data pipeline '''
    image = tf.random_uniform(shape=IMG_SIZE)
    label = tf.random_uniform(shape=[])
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return images, labels

# define the graph
images_train, labels_train = get_training_batch(BATCH_SIZE_TRAIN)
image_batch = tf.placeholder_with_default(images_train, shape=None)
label_batch = tf.placeholder_with_default(labels_train, shape=None)
new_images = tf.multiply(image_batch, -1)
new_labels = tf.multiply(label_batch, -1)

# start a session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # typical training step where batch data are drawn from the training queue
    py_images, py_labels = sess.run([new_images, new_labels])
    print('Data from queue:')
    print('Images: ', py_images)  # returned values in range [-1.0, 0.0]
    print('\nLabels: ', py_labels) # returned values [-1, 0.0]

    # typical validation step where batch data are supplied through feed_dict
    images_val = np.random.randint(0, 100, size=np.hstack((BATCH_SIZE_VAL, IMG_SIZE)))
    labels_val = np.ones(BATCH_SIZE_VAL)
    py_images, py_labels = sess.run([new_images, new_labels],
                      feed_dict={image_batch:images_val, label_batch:labels_val})
    print('\n\nData from feed_dict:')
    print('Images: ', py_images) # returned values are integers in range [-100.0, 0.0]
    print('\nLabels: ', py_labels) # returned values are -1.0

    coord.request_stop()
    coord.join(threads)