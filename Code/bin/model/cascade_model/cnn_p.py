import tensorflow as tf  
import os 
import json 

def cnn_model(word, postag, word_embedding_size, postag_embedding_size, word_embedding, postag_embedding, out_len):
    #embedding 
    with tf.name_scope('embedding'):
        word_embedding = tf.get_variable('word_embedding', initializer = word_embedding)#shape = [412183, 300])
        word_embed = tf.nn.embedding_lookup(word_embedding, word)
        postag_embedding = tf.get_variable('postag_embedding', initializer = postag_embedding)#shape = [25, 14])
        postag_embed = tf.nn.embedding_lookup(postag_embedding, postag)

    #input_representation 
    with tf.name_scope('input_representation'):
        input_representation = tf.concat([word_embed, postag_embed], 2)
        input_representation = tf.cast(input_representation, tf.float32)
        input_representation = tf.reshape(input_representation, shape = [-1, 198, 314, 1])
        
	#cnn 
    with tf.name_scope('cnn'):
        pool = []
        for kernel_size in [3, 4, 5]:
            cnn_w = tf.get_variable(str(kernel_size)+'conv_w', shape = [kernel_size, word_embedding_size[1] + postag_embedding_size[1], 1, 500])
            cnn_b = tf.get_variable(str(kernel_size)+'conv_b', shape = [500])
            cnn = tf.nn.conv2d(input_representation, cnn_w, strides = [1, 1, word_embedding_size[1] + 2 * postag_embedding_size[1], 1], padding = 'SAME')
            cnn = tf.nn.relu(tf.nn.bias_add(cnn, cnn_b))
            pooling = tf.nn.max_pool(cnn, ksize = [1, 198, 1, 1], strides = [1, 198, 1, 1], padding = 'SAME') 
            pooling = tf.reshape(pooling, [-1, 500])
            pool.append(pooling)
        pool_out = tf.concat(pool, axis = 1)

    #dense 
    with tf.name_scope('dense'):
        out = tf.layers.dense(pool_out, 800, activation = tf.nn.relu)
        out = tf.layers.dense(out, 300, activation = tf.nn.relu)	
        out = tf.layers.dense(out, out_len)

    return out

if __name__ == '__main__':
	pass 
