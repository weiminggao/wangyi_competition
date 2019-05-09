#import tensorflow as tf 
import numpy as np 
#
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
def convert_to_num_str(tags):
    num_str = []
    for i in range(len(tags)):
        if tags[i] == 3:#表示B
            start_flag = str(i)
            j = i + 1
            while j < len(tags) \
            and tags[j] != 0 \
            and tags[j] != 3:
                j += 1
            start_flag += str(j)
            num_str.append(start_flag)
    return num_str
        
#random.randrange(1,100,2)
import random
a = []
for i in range(20): 
    a.append(random.randint(0, 3))
#print(a)
#    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
a =  [0,3,2,1,0,0,0,3,1,3,3 ,2, 1, 3, 2, 2, 2, 1, 3]
print(convert_to_num_str(a))
           