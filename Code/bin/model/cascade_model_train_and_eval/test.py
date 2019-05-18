#import tensorflow as tf 
#import numpy as np 
#

###
#embedding_0 = tf.Variable(np.identity(5), dtype = tf.float32)
#embeded_0 = tf.nn.embedding_lookup(embedding_0, [[1,2,4], [2, 0, 4]])
#
#embedding_1 = tf.Variable([[1.0,2.0,3],[2.0,3.0,4.0],[5,6,7]])
#embeded_1 = tf.nn.embedding_lookup(embedding_1, [[0,0,1], [1,2,3]])
#
#input_representation = tf.concat([embeded_0, embeded_1], 2)
#with tf.Session() as sess:
#	sess.run(tf.global_variables_initializer())
#	print(sess.run(embeded_0))
#	print(sess.run(embeded_1))
#	print(sess.run(input_representation))

#a = tf.Variable([[[1,2,3], [4,5,6]],[[1,2,3],[4,5,6],[7,8,9]]])
#a = tf.reduce_mean(a, axis = 1)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(a))

#def iter():
#	a = 8
#	for i in range(5):
#		yield a + i
#
#it = iter()
#for i in range(6):
#	try:
#		ele = it.__next__()
#		while ele:
#			print(ele)
#			ele = it.__next__()
#	except:
#		it = iter()

#a = tf.constant([[[2,3,4], [5,6,7]], [[1,2,3], [2,4,6]], [[4,6,8], [3,5,7]]], dtype = tf.float32)
#a = tf.constant([[[2, 9, 10],[3, 10, 12],[0, 2, 7]], [[4, 3, 3],[5, 4, 4],[6, 4, 6]]], dtype = tf.float32)
#a = tf.reshape(a, [-1, 3])
#b = tf.constant([[1], [2], [3]], dtype = tf.float32)
#b = tf.Variable([[[100], [109]],[[2], [24]],[[3], [5]]], dtype = tf.float32)
#b = tf.transpose(tf.expand_dims(b, 1), [0, 2, 1])
#bias = tf.constant([2])
#c = b + bias
#X = np.random.randn(3, 6, 4)
#X = tf.constant([[[1,2,2,4],[5,6,7,8],[1,2,5,4],[5,6,7,8],[1,3,5,7],[3,5,7,9]],
#              [[2,4,6,2],[6,2,3,1],[8,1,7,6],[2,6,7,9],[2,8,1,9],[0,2,3,4]],
#              [[3,2,1,7],[5,1,2,9],[1,2,4,9],[2,7,1,6],[2,7,1,3],[6,3,1,8.]]])
#Y = np.array([[1,2,3,2,2,1],
#              [0,1,2,1,3,3],
#              [2,2,1,2,2,0]])
#Y = tf.placeholder(shape = [3, 6], dtype = tf.int32)
#Y = np.array([[3,3,1,3,1,3],
#              [3,3,1,2,2,2],
#              [3,3,2,1,3,1]])
#print(X)
#cell_fw = tf.contrib.rnn.LSTMCell(4)
#cell_bw = tf.contrib.rnn.LSTMCell(5)
#(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, \
#                                                            sequence_length = [6, 4, 6], initial_state_fw = cell_fw.zero_state(batch_size = 3, dtype = tf.float64), \
#                                                                    initial_state_bw = cell_bw.zero_state(batch_size = 3, dtype = tf.float64))
#outputs, _ = tf.nn.dynamic_rnn(cell = cell_fw, sequence_length = [6, 4, 6], inputs = X.astype(np.float32), initial_state = cell_fw.zero_state(batch_size = 3, dtype = tf.float32))
#log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(X, \
#                                                                      Y.astype(np.int32), \
#                                                                      sequence_lengths = np.array([6, 4, 6], dtype = np.int32))
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    print(sess.run(transition_params))
#viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(X, transition_params, sequence_length = np.array([6, 4, 6], dtype = np.int32))
#error = tf.reduce_mean(-log_likelihood)
#train_step = tf.train.AdamOptimizer(0.01).minimize(error)
#log_likelihood1, transition_params1 = tf.contrib.crf.crf_log_likelihood(X.astype(np.float32), \
#                                                                        Y.astype(np.int32), \
#                                                                        sequence_lengths = np.array([1, 1, 1], \
#                                                                                                    dtype = np.int32))
#error = tf.reduce_mean(-log_likelihood)
#mask = tf.sequence_mask([1, 3, 4])
#sequence_lengths = tf.placeholder(dtype = tf.int32, shape = [None])
#loss = tf.constant([[1, 3,4,5],[2,3,4,5],[8,10,1,10]])
#index = tf.reduce_max(sequence_lengths)
#loss = tf.reshape(loss, shape = [-1, index])
#losses = tf.boolean_mask(loss, mask)
#a = tf.placeholder(tf.int32, [None])
#b = np.zeros(10)
#b[1] = 10
#print(b)
#a = tf.placeholder(tf.int32)
#for n in tf.get_default_graph().as_graph_def().node:
#    print(n.name)
#print(transition_params.name)
#a =  tf.constant([[1,2,3,4],[5,6,7,8]], dtype = tf.float32)
#b = tf.nn.softmax(a)
##print(b.name)
#saver = tf.train.Saver(max_to_keep = 10)
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
##    print(sess.run(a))
###    print(sess.run(output_fw))
###    print(sess.run(output_bw))
###    print(sess.run(viterbi_sequence))
###    print(sess.run(viterbi_score))
#    print(sess.run(transition_params))
#    sess.run(train_step)
#    print(sess.run(transition_params))
##    for n in tf.get_default_graph().as_graph_def().node:
##        print(n.name)
#    saver.save(sess,'./test_model.ckpt')
    
#tf.reset_default_graph()
#tf.train.import_meta_graph('./test_model.ckpt.meta')
#saver=tf.train.Saver()
#with tf.Session() as sess:
#    saver.restore(sess, './test_model.ckpt')
#    for n in tf.get_default_graph().as_graph_def().node:
#        print(n.name)
#    print(sess.run(transition_params))
#    print(sess.run(tf.get_default_graph().get_tensor_by_name('log_likelihood:0')))
#def convert_to_num_str(tags):
#    num_str = []
#    for i in range(len(tags)):
#        if tags[i] == 3:#表示B
#            start_flag = str(i)
#            j = i + 1
#            while j < len(tags) \
#            and tags[j] != 0 \
#            and tags[j] != 3:
#                j += 1
#            start_flag += str(j)
#            num_str.append(start_flag)
#    return num_str
        
#random.randrange(1,100,2)
#import random
#a = []
#for i in range(20): 
#    a.append(random.randint(0, 3))
#print(a)
#           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#pso_pred =  [3,3,2,1,0,0,0,3,1,3,3 ,2, 1, 3, 2, 2, 2, 1, 0, 3, 3]
#i = 0
#words_position = []
#while i < len(pso_pred):
#    if pso_pred[i] == 3:
#        word_position = []
#        word_position.append(i)
#        i = i + 1
#        while i < len(pso_pred) and pso_pred[i] != 3 and pso_pred[i] != 0:
#            i = i + 1
#        word_position.append(i)
#        words_position.append(word_position)
#    else:
#        i +=1
#print(words_position)
#print(convert_to_num_str(a))

#import tensorflow as tf
##a = tf.Variable([111,0,0,0])
##b = tf.Variable([222,0,0,0])
##saver = tf.train.Saver()
##with tf.Session() as sess:
##    sess.run(tf.initialize_all_variables())
##    saver.save(sess, './test.ckpt1')
#g1 = tf.Graph()
#g2 = tf.Graph()
##g3 = tf.Graph()
#with g1.as_default():
#    a = tf.Variable([0,0,0,0])
#    b = tf.Variable([0,0,0,0])
#    saver = tf.train.Saver()
#    init_op = tf.initialize_all_variables()
#    sess = tf.Session(graph = g1)
#    sess.run(init_op)
###    c = a + b
#with g2.as_default():
#    a1 = tf.Variable([110,0,0,0])
#    b1 = tf.Variable([120,0,0,0])
#    saver1 = tf.train.Saver()
#    init_op1 = tf.initialize_all_variables()
#    sess1 = tf.Session(graph = g2)
#    sess1.run(init_op1)
#print(sess.run(a))
#print(sess1.run(a1))
###    c = a + b
#with g3.as_default():
#    #a = tf.constant([1,2,3,4])
#    a = tf.Variable([110,0,0,0], name = 'c1')
#    b = tf.Variable([120,0,0,0], name = 'c2')
##    c = a + b
#sess = tf.Session(graph = g1)
#sess.run(init_op)
#saver.restore(sess, './test.ckpt')
#sess1 = tf.Session(graph = g2)
#sess1.run(init_op1)
#saver1.restore(sess1, './test.ckpt1')
#for i in range(10):
#    print(sess.run(a))
#    print(sess1.run(a1))
#sess.close()
#sess1.close()
#with tf.Session(graph = g1) as sess:
##    sess.run(tf.global_variables_initializer())
###    saver.save(sess,'./test.ckpt')
#    saver.restore(sess, './test.ckpt')
##    v_names = [v.name for v in tf.all_variables()]
#    print(sess.run(a))
##    print(sess.run(c))
#with tf.Session(graph = g2) as sess:
##    sess.run(tf.initialize_all_variables())
##    v_names = [v.name for v in tf.all_variables()]
##    print(sess.run(a))
##    saver.save(sess, './test.cpkt1')
#    saver1.restore(sess, './test.ckpt1')
#    print(sess.run(a1))
#    print(sess.run(b))
#a = tf.placeholder(tf.int32)
#b = tf.placeholder(tf.int32)
#c = tf.placeholder(tf.int32)
#d = a + b
#e = c + b
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(d, feed_dict = {a:4, b:5}))
#    print(sess.run(e, feed_dict = {b:6, c:9}))
#def a(p1 = 1, p2 = 2, p3 = 4):
#    print(p1)
#    print(p2)
#    print(p3)
#c = {'p1':10, 'p2':20, 'p3':30}
#a(**c)
#import pandas as pd
##import tqdm
#data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002],'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
#frame1 = pd.DataFrame(data)
#print(frame1)
#print(frame1.drop(frame1.index,inplace=True))
#for _, row in frame1.iterrows():
#    print(row)
#d = frame1.iloc[0:2, :]
#print(d)
#print(d['pop'])
#import tensorflow as tf
#
#g1 = tf.Graph()
#g2 = tf.Graph()
#with g1.as_default():
#    a = tf.placeholder(tf.float32)
#    b = tf.placeholder(tf.float32)
#    c = a + b
#    v_names = [v.name for v in tf.all_variables()]
#    print(v_names)
#with g2.as_default():
#    a1 = tf.placeholder(tf.float32)
#    b1 = tf.placeholder(tf.float32)
#    c1 = a + b
#    v_names = [v.name for v in tf.all_variables()]
#    print(v_names)
#with tf.Session(graph = g1) as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(c, feed_dict = {a:1, b:2}))
#def a(b, c, d):
#    print(b)
#    print(c)
#    print(d)
#f = ((1,2,3),(4,5,6))
##a(*f[0])
#import tensorflow as tf
#a = tf.constant([[[1,2,3], [4,5,6]], 
#                 [[5,6,7], [8,9,7]]])
#b = tf.constant([[[1,2,3], [4,5,6]], 
#                 [[5,6,7], [8,9,7]]])
#c = tf.add(a, b)
##mask = tf.sequence_mask([1, 2])
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(c))
#    print(sess.run(mask))
#    print(sess.run(tf.boolean_mask(a, mask)))
#    print(sess.run(tf.reduce_mean_mask(a, mask)))
#import pandas as pd
#data = pd.read_json('../../../data/dev_data_pso_distinct.json', lines = True, encoding="utf8")
#print(data)
#import numpy as np
#a = np.tril(np.ones([11, 10]), -1)
#print(a)
#import pandas as pd
#a = pd.read_json('F:/wangyi_competition-master/Code/data/test_data_postag11.json', lines = True, encoding="utf8") 

#a = {}
#b = {'a':9,'b':10}
#a['l'] = []
#a['l'].append(b)
#a['l'].append(b)
#a['l'].append(b)
#with open('.\out.json', 'w', encoding='UTF-8') as f:
#    json.dump(a, f)
#import pandas as pd
#import time
#import gc
#a = pd.read_json('F:/wangyi_competition-master/Code/data/test_data_postag.json', lines = True, encoding="utf8")
#while True:
#    print('hhhhh')
#time.sleep(30)
#del a
#gc.collect()
#while True:
#    print('llll')
commit_result_f = open('./commit_result.json', encoding='UTF-8')
commit_result = commit_result_f.readlines()
commit_result1_f = open('./commit_result1.json', encoding='UTF-8')
commit_result1 = commit_result1_f .readlines()
total_result = commit_result1 + commit_result
total_resut_f = open('./total_result.json', 'w', encoding='UTF-8')
total_resut_f.writelines(total_result)
commit_result_f.close()
commit_result1_f.close()
total_resut_f.close()
