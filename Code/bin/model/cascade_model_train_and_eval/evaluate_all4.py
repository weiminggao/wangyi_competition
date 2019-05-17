# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import sys
sys.path.append('../../utils')
sys.path.append('../cascade_model')
from load_data import process_data
import cnn_p
import ps_lstmcrf
import pso_lstmcrf
import gc

model_registry = {'p':cnn_p.cnn_model, 
                  'ps':ps_lstmcrf.ps_lstmcrf_model, 
                  'pso':pso_lstmcrf.pso_lstmcrf_model}

def generate_model_placeholder_list(model_type, process_data):#测试完毕
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
    if model_type == 'p':
        return ((word_placeholder, postag_placeholder, \
                 np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                 process_data.word_embedding, process_data.postag_embedding, out_len),
                ())
    elif model_type == 'ps':
        p_placeholder = tf.placeholder(tf.int32, [None])
        sequence_lengths = tf.placeholder(tf.int32, [None])
        out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
        batch_length = tf.placeholder(tf.int32)
        return ((word_placeholder, postag_placeholder, p_placeholder, \
                 np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), np.shape(process_data.p_embedding), \
                 process_data.word_embedding, process_data.postag_embedding, process_data.p_embedding, sequence_lengths, batch_length), 
                (sequence_lengths, out_placeholder))
    elif model_type == 'pso':
        p_placeholder = tf.placeholder(tf.int32, [None])
        s_placeholder = tf.placeholder(tf.int32, [None, 10])
        sequence_lengths = tf.placeholder(tf.int32, [None])
        s_lengths = tf.placeholder(tf.int32, [None])
        out_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len])
        batch_length = tf.placeholder(tf.int32)
        return ((word_placeholder, postag_placeholder, p_placeholder, s_placeholder, \
                 np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), np.shape(process_data.p_embedding), \
                 process_data.word_embedding, process_data.postag_embedding, process_data.p_embedding, sequence_lengths, s_lengths, batch_length), \
                (sequence_lengths, out_placeholder))
    else:
        raise Exception('无效的模型')

def decorate_model(model_type, model, seq_and_out_placeholder):#测试完毕
    if model_type == 'p':
        return tf.nn.sigmoid(model)
    elif model_type == 'ps' or model_type == 'pso':
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(model, seq_and_out_placeholder[1], seq_and_out_placeholder[0])
        decode_tags, best_score = tf.contrib.crf.crf_decode(model, transition_params, seq_and_out_placeholder[0])
        return decode_tags
    else:
        raise Exception('无效的模型')
        
def generate_model_and_sess(model_type, process_data, load_path):#测试完毕
    global model_registry
    graph = tf.Graph()
    with graph.as_default():
        placeholder_list = generate_model_placeholder_list(model_type, process_data)
        model = model_registry[model_type](*placeholder_list[0])
        model = decorate_model(model_type, model, placeholder_list[1])
#        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
#        config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph = graph, config = config)
#    sess.run(init_op)
    saver.restore(sess, load_path)
    return placeholder_list[0], model, sess    

def convert_ppred_to_wordsindex(p_pred, p_dict):#测试完毕
    p_indexs = list(filter(lambda x : p_pred[x] > 0.5, range(0, len(p_pred))))
    p_words = []
    for p_index in p_indexs:
        for key, value in p_dict.items():
            if p_index == value:
                p_words.append(key)
    return p_words, p_indexs
    
def convert_psopred_to_wordsindex(pso_pred, postag, word_dict):#测试完毕
    pso_indexs = []
    pso_words = []
    words_position = []
    i = 0
    
    while i < len(pso_pred):
        if pso_pred[i] == 3:
            word_position = []
            word_position.append(i)
            i = i + 1
            while i < len(pso_pred) and pso_pred[i] != 3 and pso_pred[i] != 0:
                i = i + 1
            word_position.append(i)
            words_position.append(word_position)
        else:
            i +=1

    for word_position in words_position:
        words = ''
        pso_index = [0] * 10
        for j, position in enumerate(range(word_position[0], word_position[1])):
            words += postag[position]['word']
            if postag[position]['word'] in word_dict and j < 10:
                pso_index[j] = word_dict[postag[position]['word']]
        pso_indexs.append(pso_index)
        pso_words.append(words)
    
    return pso_words, [pso_indexs, list(map(lambda x : x[1] - x[0], words_position))]

def stats(process_data, predict_spo_lists):#测试完毕
    pred_correct_num = 0
    pred_num = 0
    real_num = 0
    print(len(process_data.valid_data))
    print(len(predict_spo_lists))
    for i, (_, rows) in enumerate(process_data.valid_data.iterrows()):
        real_num += len(rows['spo_list'])
        pred_num += len(predict_spo_lists[i]['spo_list'])
        compare_spo_list = list(map(lambda x : {'predicate':x['predicate'], \
                                                'subject':x['subject'], \
                                                'object':x['object']}, rows['spo_list']))

        for predict_spo in predict_spo_lists[i]['spo_list']:
            if predict_spo in compare_spo_list:
                pred_correct_num += 1
    precision = pred_correct_num / pred_num
    recall = pred_correct_num / real_num
    f1 = (2 * precision * recall)/(precision + recall)
    return precision, recall, f1

def commit_result(process_data, predict_spo_lists):
    schame_f = open('F:/wangyi_competition-master/Code/data/all_50_schemas', encoding='UTF-8')
    schame = {}
    line = schame_f.readline()
    while line:
        s_p_o = json.loads(line)
        if s_p_o['predicate'] not in schame:
            schame[s_p_o['predicate']] = {}
            schame[s_p_o['predicate']]['subject_type'] = s_p_o['subject_type']
            schame[s_p_o['predicate']]['object_type'] = s_p_o['object_type']
        line = schame_f.readline()
    schame_f.close()

    result_f = open('.\commit_result5.json', 'w', encoding='UTF-8')
    for i, (_, rows) in enumerate(process_data.valid_data.iterrows()):
       result = {}
       result['text'] = rows['text']
       spo_list = predict_spo_lists[i]['spo_list']
       for j in range(len(spo_list)):
           spo_list[j]['object_type'] = schame[spo_list[j]['predicate']]['object_type']
           spo_list[j]['subject_type'] = schame[spo_list[j]['predicate']]['subject_type'] 
       result['spo_list'] = spo_list
       result_f.write(json.dumps(result, ensure_ascii = False) + '\n')
    result_f.close()
    
def evaluate(p_process_data, ps_process_data, pso_process_data):#测试完毕        
    p_placeholder_list, p_model, p_sess = generate_model_and_sess('p', p_process_data, './cnn_model/cnn.ckpt2.99603e-06-43100')
    del p_process_data.word_embedding, p_process_data.postag_embedding
    gc.collect()
    
    ps_placeholder_list, ps_model, ps_sess = generate_model_and_sess('ps', ps_process_data, './lstmcrf_ps_model/lstmcrf_ps.ckpt0.0014773-72100')
    del ps_process_data.word_embedding, ps_process_data.postag_embedding
    gc.collect()
    
    pso_placeholder_list, pso_model, pso_sess = generate_model_and_sess('pso', pso_process_data, './lstmcrf_pso_model/lstmcrf_pso.ckpt0.208563-11900')#TODO
    del pso_process_data.word_embedding, pso_process_data.postag_embedding
    gc.collect()    
    
    p_test_data_iter = p_process_data.generate_batch(batch_size, p_process_data.valid_data, features = ['word_embedding', 'postag'], label_type = None)
    ps_test_data_iter = ps_process_data.generate_batch(batch_size, ps_process_data.valid_data, features = ['word_embedding', 'postag'], label_type = None)
    pso_test_data_iter = pso_process_data.generate_batch(batch_size, pso_process_data.valid_data, features = ['word_embedding', 'postag'], label_type = None)
    
    p_data, p_label = p_test_data_iter.__next__()
    ps_data, ps_label = ps_test_data_iter.__next__()
    pso_data, pso_lable = pso_test_data_iter.__next__()
    
    predict_spo_lists = []
    offset = 0
    try:
        while p_data:
            print(offset)
            p_lists = p_sess.run(p_model, feed_dict = {p_placeholder_list[0]:p_data['word_embedding'], \
                                                       p_placeholder_list[1]:p_data['postag']})
            for i, p_list in enumerate(p_lists): #i表示第i句话
                predict_spo_list = {}
                predict_spo_list['spo_list']= []
                p_words, p_indexs = convert_ppred_to_wordsindex(p_list, p_process_data.p_dict)
                if len(p_indexs) == 0:
                    predict_spo_lists.append(predict_spo_list)
                    continue
                s_lists = ps_sess.run(ps_model, feed_dict = {ps_placeholder_list[0]:np.tile(ps_data['word_embedding'][i:i + 1, :], (len(p_indexs), 1)), \
                                                             ps_placeholder_list[1]:np.tile(ps_data['postag'][i:i + 1, :], (len(p_indexs), 1)), \
                                                             ps_placeholder_list[2]:p_indexs, \
                                                             ps_placeholder_list[-2]:[len(ps_process_data.valid_data.iloc[offset * batch_size + i, :]['postag'])] * len(p_indexs), \
                                                             ps_placeholder_list[-1]:len(p_indexs)})
                for j, s_list in enumerate(s_lists):    #j表示第i句话的第j个关系
                    s_words, s_indexs = convert_psopred_to_wordsindex(s_list, ps_process_data.valid_data.iloc[offset * batch_size + i, :]['postag'], pso_process_data.word_dict)
                    if len(s_indexs[0]) == 0:
                        continue
                    o_lists = pso_sess.run(pso_model, feed_dict = {pso_placeholder_list[0]:np.tile(pso_data['word_embedding'][i:i + 1, :], (len(s_indexs[0]), 1)), \
                                                                   pso_placeholder_list[1]:np.tile(pso_data['postag'][i:i + 1, :], (len(s_indexs[0]), 1)), \
                                                                   pso_placeholder_list[2]:p_indexs[j:j + 1] * len(s_indexs[0]), \
                                                                   pso_placeholder_list[3]:s_indexs[0], \
                                                                   pso_placeholder_list[-3]:[len(pso_process_data.valid_data.iloc[offset * batch_size + i, :]['postag'])] * len(s_indexs[0]), \
                                                                   pso_placeholder_list[-2]:s_indexs[1],
                                                                   pso_placeholder_list[-1]:len(s_indexs[0])})
                    for k, o_list in enumerate(o_lists):    #k表示第i句话的第j个关系的第k个s
                        o_words, o_indexs = convert_psopred_to_wordsindex(o_list, pso_process_data.valid_data.iloc[offset * batch_size + i, :]['postag'], pso_process_data.word_dict)
                        for l, o in enumerate(o_indexs[0]):     #k表示第i句话的第j个关系的第k个s的第l个o
                            spo = {}
                            spo['predicate'] = p_words[j]
                            spo['subject'] = s_words[k]
                            spo['object'] = o_words[l]
                            predict_spo_list['spo_list'].append(spo)
                predict_spo_lists.append(predict_spo_list)
            p_data, p_label = p_test_data_iter.__next__()
            ps_data, ps_label = ps_test_data_iter.__next__()
            pso_data, pso_lable = pso_test_data_iter.__next__()
            offset += 1
    except Exception as e:
        print('预测完毕')
        with open('.\out5.json', 'w', encoding='UTF-8') as f:
            out = {}
            out['predict_spo_lists'] = predict_spo_lists
            json.dump(out, f, ensure_ascii = False)
#        print(predict_spo_lists)
        
        commit_result(p_process_data, predict_spo_lists)
#        precision, recall, f1 = stats(p_process_data, predict_spo_lists)
#        print('cascade模型的precision:{}, recall:{}, f1:{}'.format(precision, recall, f1))
            
    p_sess.close()
    ps_sess.close()
    pso_sess.close()

if __name__ == '__main__':
    p_train_data_path_list = ['../../../data/train_data.json']
    p_test_data_path = '../../../data/dev_data.json'
    ps_train_data_path_list = ['../../../data/train_data_ps.json']
    ps_test_data_path = '../../../data/dev_data_ps.json'
    pso_train_data_path_list = ['../../../data/train_data_pso.json']
    pso_test_data_path = '../../../data/dev_data_pso.json'
    
    pre_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    baike_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    postag_path = '../../../data/pos'
    p_path = '../../../data/all_50_schemas'
    
    p_process_data = process_data(p_train_data_path_list, p_test_data_path, pre_word_embedding_path, \
                                  baike_word_embedding_path, postag_path, p_path, 'F:/wangyi_competition-master/Code/data/test_data_postag5.json')
    del p_process_data.train_data, p_process_data.test_data
    gc.collect()
    
    ps_process_data = process_data(ps_train_data_path_list, ps_test_data_path, pre_word_embedding_path, \
                                   baike_word_embedding_path, postag_path, p_path, 'F:/wangyi_competition-master/Code/data/test_data_postag5.json')
    del ps_process_data.train_data, ps_process_data.test_data
    gc.collect()
    
    pso_process_data = process_data(pso_train_data_path_list, pso_test_data_path, pre_word_embedding_path, \
                                    baike_word_embedding_path, postag_path, p_path, 'F:/wangyi_competition-master/Code/data/test_data_postag5.json')
    del pso_process_data.train_data, pso_process_data.test_data
    gc.collect()
    
    print('F:/wangyi_competition-master/Code/data/test_data_postag5.json')
    batch_size = 1
    out_len = 49
    evaluate(p_process_data, ps_process_data, pso_process_data)
