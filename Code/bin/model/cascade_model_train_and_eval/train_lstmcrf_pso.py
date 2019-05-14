# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../utils')
sys.path.append('../cascade_model')
from load_data import process_data
import pso_lstmcrf

RESTORE = False

def evaluate(process_data):
    pred_correct_num = 0
    pred_num = 0
    real_num = 0
    def stats(predict_output, label):
        def convert_to_num_str(tags):
            num_str = []
            for i in range(len(tags)):
                if tags[i] == 3:#表示B
                    start_flag = str(i)
                    j = i + 1
                    while j < len(tags) and tags[j] != 0 and tags[j] != 3:
                        j += 1
                    start_flag += str(j)
                    num_str.append(start_flag)
            return num_str
        
        nonlocal pred_correct_num, pred_num, real_num
        for i in range(len(predict_output)):
            pred_num_str = convert_to_num_str(predict_output[i])
            lab_num_str = convert_to_num_str(label[i])
            pred_num += len(pred_num_str)
            real_num += len(lab_num_str)
            for ele in pred_num_str:
                if ele in lab_num_str:
                    pred_correct_num += 1
    
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    p_placeholder = tf.placeholder(tf.int32, [None])
    s_placeholder = tf.placeholder(tf.int32, [None, 5])
    sequence_lengths = tf.placeholder(tf.int32, [None])
    out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
    batch_length = tf.placeholder(tf.int32)
    
    pso_model = pso_lstmcrf.pso_lstmcrf_model(word_placeholder, postag_placeholder, p_placeholder, s_placeholder, \
                                              np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                              np.shape(process_data.p_embedding), process_data.word_embedding, process_data.postag_embedding, \
                                              process_data.p_embedding, sequence_lengths, batch_length)
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(pso_model, out_placeholder, sequence_lengths)
    decode_tags, best_score = tf.contrib.crf.crf_decode(pso_model, transition_params, sequence_lengths)
    test_data_iter = process_data.generate_batch(batch_size, process_data.test_data, \
                                                 features = ['word_embedding', 'postag', 'p', 's', 'sequence_lengths'], label_type = 'o')
    saver = tf.train.Saver()	
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('../cascade_model/lstmcrf_pso_model'))
#        saver.restore(sess, './lstmcrf_ps_model\lstmcrf_pso.ckpt0.0014773-72100')
        decode_tags, best_score = tf.contrib.crf.crf_decode(pso_model, transition_params, sequence_lengths)
        try:
            data, label = test_data_iter.__next__()
            while data:
                _decode_tags = sess.run(decode_tags, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], \
                                                                  p_placeholder:data['p'], s_placeholder:data['s'],\
                                                                  sequence_lengths:data['sequence_lengths'], batch_length:len(data['sequence_lengths'])})		
                stats(_decode_tags, label)
                data, label = test_data_iter.__next__()
        except Exception as e:
            print('预测完毕')
            precision = pred_correct_num / pred_num
            recall = pred_correct_num / real_num
            f1 = (2 * precision * recall)/(precision + recall)
            print('pred_correct_num:{}, pred_num:{}, real_num:{}'.format(pred_correct_num, pred_num, real_num))
            print('precision:{},recall:{},f1:{}'.format(precision, recall, f1))
    return f1, precision, recall
        
def train(learning_rate, batch_size, epoch, process_data):
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    p_placeholder = tf.placeholder(tf.int32, [None])
    s_placeholder = tf.placeholder(tf.int32, [None, 5])
    sequence_lengths = tf.placeholder(tf.int32, [None])
    batch_length = tf.placeholder(tf.int32)
    out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
    
    pso_model = pso_lstmcrf.pso_lstmcrf_model(word_placeholder, postag_placeholder, p_placeholder, s_placeholder, \
                                             np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                             np.shape(process_data.p_embedding), process_data.word_embedding, \
                                             process_data.postag_embedding, process_data.p_embedding, \
                                             sequence_lengths, batch_length)
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(pso_model, out_placeholder, sequence_lengths)
    error = tf.reduce_mean(-log_likelihood)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)
    
    train_data_iter = process_data.generate_batch(batch_size, process_data.train_data, \
                                                  features = ['word_embedding', 'postag', 'p', 's', 'sequence_lengths'], label_type = 'o')
    saver = tf.train.Saver(max_to_keep = 10)	
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if RESTORE:
            saver.restore(sess, tf.train.latest_checkpoint('../cascade_model/lstmcrf_pso_model'))
        
        step = 0
        for ep in range(epoch):
            print('epoch:{}'.format(ep))
            try:
                data, label = train_data_iter.__next__()
                while data:
                    step += 1
                    sess.run(train_step, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], \
                                                      p_placeholder:data['p'], s_placeholder:data['s'], \
                                                      sequence_lengths:data['sequence_lengths'], batch_length:len(data['sequence_lengths']), out_placeholder:label})		
                    _error = sess.run(error, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], \
                                                          p_placeholder:data['p'], s_placeholder:data['s'], \
                                                          sequence_lengths:data['sequence_lengths'], batch_length:len(data['sequence_lengths']), out_placeholder:label})		
                    data, label = train_data_iter.__next__()
                    if step % 100 == 0:
                        print(step)
                        saver.save(sess, '../cascade_model/lstmcrf_pso_model/lstmcrf_pso.ckpt'+str(_error), global_step = step, write_meta_graph=False)
                        print('step:{}, error:{}'.format(step, _error))
            except Exception as e:
                print(e)
                train_data_iter = process_data.generate_batch(batch_size, process_data.train_data,  features = ['word_embedding', 'postag', 'p', 's', 'sequence_lengths'], label_type = 'o')		

if __name__ == '__main__':	
    train_data_path_list = ['../../../data/train_data_pso.json']
    test_data_path = '../../../data/dev_data_pso.json'
    pre_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    baike_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    postag_path = '../../../data/pos'
    p_path = '../../../data/all_50_schemas'
    process_data = process_data(train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path)
    
    batch_size = 128
    learning_rate = 0.001 #0.01 #0.001    #0.0000001收敛较慢
    epoch = 100
    train(learning_rate, batch_size, epoch, process_data)
#    evaluate(process_data)
