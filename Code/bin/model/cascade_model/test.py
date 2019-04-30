#import tensorflow as tf 
#import numpy as np 
'''
embedding_0 = tf.Variable(np.identity(5), dtype = tf.float32)
embeded_0 = tf.nn.embedding_lookup(embedding_0, [[1,2,4], [2,3,4]])

embedding_1 = tf.Variable([[1.0,2.0,3],[2.0,3.0,4.0],[5,6,7]])
embeded_1 = tf.nn.embedding_lookup(embedding_1, [[0,0,1], [1,2,3]])

input_representation = tf.concat([embeded_0, embeded_1], 2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(embeded_0))
	print(sess.run(embeded_1))
	print(sess.run(input_representation))
'''
def iter():
	a = 8
	for i in range(5):
		yield a + i

it = iter()
for i in range(6):
	try:
		ele = it.__next__()
		while ele:
			print(ele)
			ele = it.__next__()
	except:
		it = iter()
