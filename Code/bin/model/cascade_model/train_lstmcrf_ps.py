# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../utils')
from load_data import process_data
import ps_lstmcrf

RESTORE = False

def evaluate(process_data):
    pass
        
def train(learning_rate, batch_size, epoch, process_data):
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    p_placeholder = tf.placeholder(tf.int32, [None])
    sequence_lengths = tf.placeholder(tf.int32, [None])
    out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
    
    ps_model = ps_lstmcrf.ps_lstmcrf_model(word_placeholder, postag_placeholder, p_placeholder, np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                           np.shape(process_data.p_embedding), process_data.word_embedding, process_data.postag_embedding, process_data.p_embedding, sequence_lengths, batch_size)
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(ps_model, out_placeholder, sequence_lengths)
    error = tf.reduce_mean(-log_likelihood)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)
    
    train_data_iter = process_data.generate_batch(batch_size, process_data.train_data,  \
                                                  features = ['word_embedding', 'postag', 'p', 'sequence_lengths'], label_type = 's')
    saver = tf.train.Saver(max_to_keep = 10)	
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if RESTORE:
            saver.restore(sess, tf.train.latest_checkpoint('./lstmcrf_ps_model'))
        
        step = 0
        for ep in range(epoch):
            print('epoch:{}'.format(ep))
            try:
                data, label = train_data_iter.__next__()
                while data:
                    step += 1
                    sess.run(train_step, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], \
                                                      p_placeholder:data['p'], sequence_lengths:data['sequence_lengths'], out_placeholder:label})		
                    _error = sess.run(error, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], \
                                                          p_placeholder:data['p'], sequence_lengths:data['sequence_lengths'], out_placeholder:label})		
                    data, label = train_data_iter.__next__()
                    if step % 100 == 0:
                        print(step)
                        saver.save(sess, './lstmcrf_ps_model/lstmcrf_ps.ckpt'+str(_error), global_step = step, write_meta_graph=False)
                        print('step:{}, error:{}'.format(step, _error))
            except Exception as e:
                print(e)
                train_data_iter = process_data.generate_batch(batch_size, process_data.train_data,  features = ['word_embedding', 'postag', 'p'], label_type = 's')		

if __name__ == '__main__':	
    train_data_path_list = ['../../../data/train_data_ps.json']
    test_data_path = '../../../data/dev_data_ps.json'
    pre_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    baike_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    postag_path = '../../../data/pos'
    p_path = '../../../data/all_50_schemas'
    process_data = process_data(train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path)
    
    batch_size = 128
    learning_rate = 0.0001    #0.0000001收敛较慢
    epoch = 100
    train(learning_rate, batch_size, epoch, process_data)