import tensorflow as tf 
import os 
import json 

def att_cnn_model(word, pos, relation, word_embedding_size, pos_embedding_size, relation_embedding_size):
	#embedding 
	with tf.name_scope('embedding'):
		word_embedding = tf.get_variable('word_embedding', shape = word_embedding_size)
		word_embed = tf.nn.embedding_lookup(word_embedding, word)
		pos_embedding_0 = tf.get_variable('pos_embedding_0', shape = pos_embedding_size)
		pos_embed_0 = tf.nn.embedding_lookup(pos_embedding_0, pos[0])
		pos_embedding_1 = tf.get_variable('pos_embedding_1', shape = pos_embedding_size)
		pos_embed_1 = tf.nn.embedding_lookup(pos_embedding_1, pos[1])
		relation_embedding = tf.get_variable('relation_embedding', shape = relation_embedding_size)
		relation_embed = tf.nn.embedding_lookup(relation_embedding, relation)

	#input_representation 
	with tf.name_scope('input_representation'):
		input_representation = tf.concat([word_embedding, pos_embedding_0, pos_embedding_1], 2)
	
	#attention_diag
	with tf.name_scope('attention'):	 
		attention_m = tf.get_variable('attention_M', shape = [word_embedding_size[1] + 2 * pos_embedding_size[1]], relation_embedding_size[1])
		attention_m = tf.tile(tf.expand_dims(attention_m, 0), [len(word), 1, 1])		
		wm = tf.matmul(input_representation, attention_m)		
		r_transpose = tf.transpose(tf.expand_dims(relation_embed, 1), [0, 2, 1])
		wmr = tf.matmul(wm, r_transpose)
		bias = tf.get_variable('bias', shape = [1])
		wmr = wmr + bias
		diag = tf.reshape(wmr, [len(word), -1])
		diag = tf.nn.softmax(diag)
		diag = tf.reshape(diag, [len(word), -1, 1])

	#1d cnn
	with tf.name_scope('cnn'): 		
		cnn_input = tf.multiply(input_representation, diag)
		cnn1 = tf.layers.conv1d(cnn_input, filters = 1024, kernel_size = 3, strides = 1, avtivation = 'tanh')	
		pooling = tf.layers.max_pooling1d(cnn1, pool_size = word_embedding_size[1] + 2 * pos_embedding_size[1] - 2, strides = 1)
	
	#score
	with tf.name_scope('score'):
		score_input = tf.layers.flatten(pooling)
		u = tf.get_variable('u', shape = [1000, relation_embedding_size[1]])
		u = tf.tile(tf.expand_dims(u, 0), [len(word), 1, 1])
		score = tf.matmul(score_input, u, relation_embed)
		score_output = tf.nn.sigmoid(score)	
	return score_output 
	
if __name__ == '__main__':
	pass
