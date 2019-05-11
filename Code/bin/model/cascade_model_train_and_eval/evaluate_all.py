# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../utils')
sys.path.append('../cascade_model')
from load_data import process_data
import cnn_p
import ps_lstmcrf
import pso_lstmcrf

model_registry = {'p':cnn_p.cnn_model, 
                  'ps':ps_lstmcrf.ps_lstmcrf_model, 
                  'pso':pso_lstmcrf.pso_lstmcrf_model}

def decorator_model(model_type, model, seq_and_out_placeholder):
    if model_type == 'p':
        return tf.nn.sigmoid(model)
    elif model_type == 'ps' or model_type == 'pso':
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(model, seq_and_out_placeholder[1], seq_and_out_placeholder[0])
        decode_tags, best_score = tf.contrib.crf.crf_decode(model, transition_params, seq_and_out_placeholder[0])
        return decode_tags
    else:
        raise '无效的模型'
        
def generate_model_and_sess(model_type, placeholder_list, load_path):
    global model_registry
    graph = tf.Graph()
    with graph.as_graph_def():
        model = model_registry[model_type](*placeholder_list[0])
        model = decorator_model(model_type, model, placeholder_list[1])
        saver = tf.train.Saver()
    sess = tf.Session(graph)
    saver.restore(sess, load_path)
    return model, sess    

def convert_pspred_to_wordindex():
    pass
 
def stats(process_data, predict_spo_lists):
    pass
    
def evaluate(process_data):    
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    p_placeholder = tf.placeholder(tf.int32, [None])
    s_placeholder = tf.placeholder(tf.int32, [None, 5])
    sequence_lengths = tf.placeholder(tf.int32, [None])
    out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
    batch_length = tf.placeholder(tf.int32)
    
    p_model, p_sess = generate_model_and_sess('p', ((word_placeholder, postag_placeholder, \
                                                    np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                                    process_data.word_embedding, process_data.postag_embedding, out_len)), \
                                              './cnn_model\cnn.ckpt2.99603e-06-43100')
    ps_model, ps_sess = generate_model_and_sess('ps', ((word_placeholder, postag_placeholder, p_placeholder, 
                                                       np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), np.shape(process_data.p_embedding), \
                                                       process_data.word_embedding, process_data.postag_embedding, process_data.p_embedding, sequence_lengths, batch_length), 
                                                       (sequence_lengths, out_placeholder)), \
                                                './lstmcrf_ps_model\lstmcrf_ps.ckpt0.0014773-72100')
    pso_model, pso_sess = generate_model_and_sess('pso', ((word_placeholder, postag_placeholder, p_placeholder, s_placeholder, \
                                                          np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), np.shape(process_data.p_embedding), \
                                                          process_data.word_embedding, process_data.postag_embedding, process_data.p_embedding, sequence_lengths, batch_length), \
                                                          (sequence_lengths, out_placeholder)), \
                                                  './lstmcrf_pso_model/')#TODO
        
    test_data_iter = process_data.generate_batch(batch_size, process_data.test_data, \
                                                 features = ['word_embedding', 'postag'], label_type = 'p')
    offset = 0
    predict_spo_lists = []
    try:
        data, label = test_data_iter.__next__()
        p_lists = p_sess.run(p_model, feed_dict = {word_placeholder:data['word_embedding'], \
                                                   postag_placeholder:data['postag']})
        
        for i, p_list in enumerate(p_lists): #i表示第i句话
            predict_spo_list = {}
            predict_spo_list['spo_list']= []
            p_data = list(filter(lambda x : p_list[x] > 0.5, range(0, len(p_list))))
            if len(p_data) > 0:
                predict_spo_lists.append(predict_spo_list)
                continue
                
            s_lists = ps_sess.run(ps_model, feed_dict = {word_placeholder:data['word_embedding'][offset * batch_size + i:offset * batch_size + i + 1] * len(p_data), \
                                                         postag_placeholder:data['postag'][offset * batch_size + i:offset * batch_size + i + 1] * len(p_data), \
                                                         p_placeholder:p_data, \
                                                         sequence_lengths:[len(process_data.test_data.iloc[offset * batch_size + i, :]['postag'])] * len(p_data), \
                                                         batch_length:len(p_data)})
            for j, s_list in enumerate(s_lists):    #j表示第i句话的第j个关系
                s_indexs = convert_pspred_to_wordindex(s_list, process_data.test_data.iloc[offset * batch_size + i, :]['postag'])
                o_lists = pso_sess.run(pso_model, feed_dict = {word_placeholder:data['word_embedding'][offset * batch_size + i:offset * batch_size + i + 1] * len(s_indexs), \
                                                               postag_placeholder:data['postag'][offset * batch_size + i:offset * batch_size + i + 1] * len(s_indexs), \
                                                               p_placeholder:p_data[j] * len(s_indexs), \
                                                               s_placeholder:s_indexs, \
                                                               sequence_lengths:[len(process_data.test_data.iloc[offset * batch_size + i, :]['postag'])] * len(s_indexs), \
                                                               batch_length:len(s_indexs)})
                for k, o_list in enumerate(o_lists):    #k表示第i句话的第j个关系的第k个s
                    o_indexs = convert_pspred_to_wordindex(o_list, process_data.test_data.iloc[offset * batch_size + i, :]['postag'])
                    for l, o in enumerate(o_indexs):     #k表示第i句话的第j个关系的第k个s的第l个o
                        spo = {}
                        spo['predicated'] = p_data[j]
                        spo['subject'] = s_indexs[k]#TODO
                        spo['object'] = o_indexs[l]#TODO
                        predict_spo_list['spo_list'].append(spo)
            predict_spo_lists.append(predict_spo_list)
        offset += 1
    except Exception as e:
        print('预测完毕')
        precision, recall, f1 = stats(process_data, predict_spo_lists)
        print('cascade模型的precision:{}, recall:{}, f2:{}'.format(precision, recall, f1))
            
    p_sess.close()
    ps_sess.close()
    pso_sess.close()
    

if __name__ == '__main__':
    train_data_path_list = ['../../../data/train_data.json']
    test_data_path = '../../../data/dev_data.json'
    pre_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    baike_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    postag_path = '../../../data/pos'
    p_path = '../../../data/all_50_schemas'
    process_data = process_data(train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path)
    
    batch_size = 256
    out_len = 49
    evaluate(process_data)