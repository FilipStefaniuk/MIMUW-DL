import os
import random
import tensorflow as tf

def input_parser(img_path, label_path, size):

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    label_file = tf.read_file(label_path)
    label_decoded = tf.image.decode_png(label_file)

    last_img_dim = tf.shape(img_decoded)[-1]
    last_label_dim = tf.shape(label_decoded)[-1]

    conc = tf.concat([img_decoded, label_decoded], axis=2)
    cropped = tf.random_crop(conc, size=tf.concat([size, [last_img_dim + last_label_dim]], axis=0))

    return cropped[:, :, :last_img_dim], cropped[:, :, last_img_dim:]

if __name__ == '__main__':

    random.seed(42)

    file_names = [os.path.splitext(filename)[0] for filename in os.listdir('/scidata/assignment2/training/images/')]
    random.shuffle(file_names)
    validation_file_names = file_names[:(len(file_names)*20//100)]
    training_file_names = file_names[(len(file_names)*20//100):]
    
    val_img_paths = [os.path.join('/scidata/assignment2/training/images', filename +  '.jpg') for filename in validation_file_names]
    tr_img_paths = [os.path.join('/scidata/assignment2/training/images', filename + '.jpg') for filename in training_file_names]
    val_lb_paths = [os.path.join('/scidata/assignment2/training/labels_plain', filename + '.png') for filename in validation_file_names]
    tr_lb_paths = [os.path.join('/scidata/assignment2/training/labels_plain', filename + '.png') for filename in training_file_names]

    assert all(map(os.path.isfile, val_img_paths))
    assert all(map(os.path.isfile, tr_img_paths))
    assert all(map(os.path.isfile, val_lb_paths))
    assert all(map(os.path.isfile, tr_lb_paths))

    train_imgs = tf.constant(tr_img_paths)
    train_labels = tf.constant(tr_lb_paths)
    val_imgs = tf.constant(val_img_paths)
    val_labels = tf.constant(val_lb_paths)

    crop_size = tf.constant([512, 512])

    tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    tr_data = tr_data.apply(tf.contrib.data.shuffle_and_repeat(64))
    tr_data = tr_data.map(lambda x, y: input_parser(x, y, crop_size))
    tr_data = tr_data.batch(32)

    # val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
    # val_data = val_data.map(input_parser)

    iterator = tr_data.make_one_shot_iterator()
    next_batch = iterator.get_next()

    x = tf.placeholder(tf.float32, [None, 512, 512, 3], name="input")
    x_img = tf.summary.image('input', x, max_outputs=10)

    y = tf.placeholder(tf.float32, [None, 512, 512, 1], name="label")
    y_img = tf.summary.image('label', y, max_outputs=10)

    writer = tf.summary.FileWriter('summary')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # writer.add_graph(sess.graph)
        
        imgs, labels = sess.run(next_batch)
        # summary, _ = sess.run(x_img, x, feed_dict={x: imgs})
        
        # sess.run(x, feed_dict={x: imgs})
        summary1 = sess.run(x_img, feed_dict={x: imgs})
        summary2 = sess.run(y_img, feed_dict={y: labels})

        writer.add_summary(summary1)
        writer.add_summary(summary2)