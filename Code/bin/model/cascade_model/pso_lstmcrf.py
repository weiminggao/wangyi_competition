# -*- coding: utf-8 -*-
import tensorflow as tf
def pso_lstmcrf_model(word, postag, p, s, word_embedding_size, postag_embedding_size, p_embedding_size, \
                     word_embedding, postag_embedding, p_embedding, sequence_lengths, batch_size):
    #embedding
    with tf.name_scope('embedding'):
        word_embedding = tf.get_variable('word_embedding', initializer = word_embedding)#shape = [412183, 300])
        word_embed = tf.nn.embedding_lookup(word_embedding, word)
        postag_embedding = tf.get_variable('postag_embedding', initializer = postag_embedding)#shape = [25, 14])
        postag_embed = tf.nn.embedding_lookup(postag_embedding, postag)
        p_embedding = tf.get_variable('p_embedding', initializer = p_embedding)
        p_embed = tf.nn.embedding_lookup(p_embedding, p)
        s_embed = tf.nn.embedding_lookup(word_embedding, s)
        s_embed = tf.reduce_mean(s_embed, axis = 1)

    #inpuy_representation
    with tf.name_scope('input_representation'):
        input_representation = tf.concat([word_embed, postag_embed], 2)
        input_representation = tf.cast(input_representation, tf.float32)
        
    #p_attention_diag
    with tf.name_scope('p_attention'):	 
        p_attention_m = tf.get_variable('p_attention_M', shape = [word_embedding_size[1] + postag_embedding_size[1], p_embedding_size[1]])
        p_attention_m = tf.tile(tf.expand_dims(p_attention_m, 0), [batch_size, 1, 1])		
        p_wm = tf.matmul(input_representation, p_attention_m)		
        p_transpose = tf.transpose(tf.expand_dims(p_embed, 1), [0, 2, 1])
        p_wmr = tf.matmul(p_wm, p_transpose)
        p_bias = tf.get_variable('p_bias', shape = [1])
        p_wmr = p_wmr + p_bias#加法是在最后一维上加
        p_diag = tf.reshape(p_wmr, [batch_size, -1])
        p_diag = tf.nn.softmax(p_diag)
        p_diag = tf.reshape(p_diag, [batch_size, 198, -1])
        
    #s_attension_diag
    with tf.name_scope('s_attention'):
        s_input_representation = tf.multiply(input_representation, p_diag)
        s_attention_m = tf.get_variable('s_attention_M', shape = [word_embedding_size[1] + postag_embedding_size[1], word_embedding_size[1]])
        s_attention_m = tf.tile(tf.expand_dims(s_attention_m, 0), [batch_size, 1, 1])		
        s_wm = tf.matmul(s_input_representation, s_attention_m)		
        s_transpose = tf.transpose(tf.expand_dims(s_embed, 1), [0, 2, 1])
        s_wmr = tf.matmul(s_wm, s_transpose)
        s_bias = tf.get_variable('s_bias', shape = [1])
        s_wmr = s_wmr + s_bias
        s_diag = tf.reshape(s_wmr, [batch_size, -1])
        s_diag = tf.nn.softmax(s_diag)
        s_diag = tf.reshape(s_diag, [batch_size, 198, 1])
    
    #bi_lstm
    with tf.name_scope('bi_lstm'):
        lstm_input = tf.multiply(s_input_representation, s_diag)
        cell_fw = tf.contrib.rnn.LSTMCell(512)
        cell_bw = tf.contrib.rnn.LSTMCell(512)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, lstm_input, \
                                                                    sequence_length = sequence_lengths, dtype = tf.float32)
#                                                                    initial_state_fw = cell_fw.zero_state(batch_size = batch_size, dtype = tf.float32), \
#                                                                    initial_state_bw = cell_bw.zero_state(batch_size = batch_size, dtype = tf.float32))
        context = tf.concat([output_fw, output_bw], axis = -1)
        context = tf.reshape(context, [-1, 1024])
        
    #crf_decode
    with tf.name_scope('output'):
        w = tf.get_variable('w', shape = [1024, 4], dtype = tf.float32)
        b = tf.get_variable('b', shape = [4], dtype = tf.float32)
        prediction = tf.matmul(context, w) + b
        prediction = tf.reshape(prediction, shape = [-1, 198, 4])
    return prediction