import tensorflow as tf
import os
import json
import numpy as np
import sys
sys.path.append('../../utils')
sys.path.append('../cascade_model')
from load_data import process_data
import cnn_p

RESTORE = True

def evaluate(process_data):
    pred_correct_num = 0
    pred_num = 0
    real_num = 0
    def stats(predict_output, label):
        nonlocal pred_correct_num, pred_num, real_num
        real_num += sum(list(map(lambda x : sum(x), label)))
        for i, row in enumerate(predict_output):
            for j, _ in enumerate(row):
                if predict_output[i][j] > 0.5:
                    pred_num += 1
                    if label[i][j] == 1:
                        pred_correct_num += 1
    
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    out_placeholder = tf.placeholder(tf.float32, [None, out_len]) 
    cnn_model = cnn_p.cnn_model(word_placeholder, postag_placeholder, \
                                np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                process_data.word_embedding, process_data.postag_embedding, out_len)
    test_data_iter = process_data.generate_batch(batch_size, process_data.test_data) 
    saver = tf.train.Saver()    
    with tf.Session() as sess:
        saver.restore(sess, './cnn_model\cnn.ckpt2.99603e-06-43100')#tf.train.latest_checkpoint('./cnn_model'))
        try:
            data, label = test_data_iter.__next__()
            while data:
                predict_output = sess.run(cnn_model, feed_dict = {word_placeholder:data['word_embedding'], \
                                                                postag_placeholder:data['postag'], \
                                                                out_placeholder:label})
                predict_output = sess.run(tf.nn.sigmoid(predict_output))
                stats(predict_output, label)
                data, label = test_data_iter.__next__()
        except:
            print('预测完毕')
            precision = pred_correct_num / pred_num
            recall = pred_correct_num / real_num
            f1 = (2 * precision * recall)/(precision + recall)
            print('pred_correct_num:{}, pred_num:{}, real_num:{}'.format(pred_correct_num, pred_num, real_num))
            print('precision:{},recall:{},f1:{}'.format(precision, recall, f1))
    return f1, precision, recall
        
def train(learning_rate, batch_size, epoch, out_len, process_data): 
    word_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    postag_placeholder = tf.placeholder(tf.int32, [None, process_data.max_len]) 
    out_placeholder = tf.placeholder(tf.float32, [None, out_len]) 
    cnn_model = cnn_p.cnn_model(word_placeholder, postag_placeholder, \
                                np.shape(process_data.word_embedding), np.shape(process_data.postag_embedding), \
                                process_data.word_embedding, process_data.postag_embedding, out_len) 
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = out_placeholder, logits = cnn_model))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

    train_data_iter = process_data.generate_batch(batch_size, process_data.train_data)
    saver = tf.train.Saver(max_to_keep = 10)	
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if RESTORE:
            saver.restore(sess, tf.train.latest_checkpoint('./cnn_model'))
        
        step = 0
        for ep in range(epoch):
            print('epoch:{}'.format(ep))
            try:
                data, label = train_data_iter.__next__()
                while data:
                    step += 1
                    sess.run(train_step, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], out_placeholder:label})		
                    _error = sess.run(error, feed_dict = {word_placeholder:data['word_embedding'], postag_placeholder:data['postag'], out_placeholder:label})		
                    data, label = train_data_iter.__next__()
                    if step % 100 == 0:
                        print(step)
                        saver.save(sess, './cnn_model/cnn.ckpt'+str(_error), global_step = step, write_meta_graph=False)
                        print('step:{}, error:{}'.format(step, _error))
            except Exception as e:
                print(e)
                train_data_iter = process_data.generate_batch(batch_size, process_data.train_data)		

if __name__ == '__main__':	
    train_data_path_list = ['../../../data/train_data.json']
    test_data_path = '../../../data/dev_data.json'
    pre_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    baike_word_embedding_path = '../../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
    postag_path = '../../../data/pos'
    p_path = '../../../data/all_50_schemas'
    process_data = process_data(train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path)
    
    batch_size = 256
    learning_rate = 0.0001    #0.0000001收敛较慢
    out_len = 49
    epoch = 100
#    train(learning_rate, batch_size, epoch, out_len, process_data)
    evaluate(process_data)