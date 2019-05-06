import pandas as pd 
import json 
import numpy as np 
from tqdm import tqdm 

class process_data(object):
    def __init__(self, train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path): 
        print('初始化开始')
        self.train_data_path_list = train_data_path_list 
        self.test_data_path = test_data_path 
        self.pre_word_embedding_path = pre_word_embedding_path 
        self.baike_word_embedding_path = baike_word_embedding_path 
        self.postag_path = postag_path 
        self.p_path = p_path 
        self.train_data = pd.read_json(self.train_data_path_list[0], lines = True, encoding="utf8") 
        for path in train_data_path_list[1:]: 
            self.train_data = pd.concat(self.train_data, pd.read_json(path, lines = True, encoding="utf8"))
        self.test_data = pd.read_json(self.test_data_path, lines = True, encoding="utf8") 
        self.max_len, self.word_dict = self.generate_maxlen_and_worddict() 
        self.p_dict = self.generate_p_dict()
        self.postag_dict = self.generate_postag_dict()
        self.postag_embedding = self.generate_postag_embedding() 
        self.word_embedding = self.generate_word_embedding()
        self.feature_func_dict = {'word_embedding':self.parse_word_embedding, 'pos_embedding':self.parse_pos_embedding, 'postag':self.parse_postag}
        print('初始化完成')

    def save_word_embedding(self, saved_path, word_embedding):
        f = open(saved_path, 'w', encoding='UTF-8')
        def get_key(value):
            for k, v in self.word_dict.items():
                if v == value:
                    return k
                raise '找不到对应的word'
        for i, embedding in enumerate(word_embedding):
            f.write(get_key(i) + '\t' + ','.join(list(map(lambda x:str(x), embedding))) + '\n')
        f.close()		
	
    def generate_p_dict(self):#测试完毕 
        f = open(self.p_path, encoding='UTF-8')
        p = f.readline()
        p_dict = {}
        count = 0
        while p:
            predicate = json.loads(p)['predicate']
            if predicate not in p_dict: 
                p_dict[predicate] = count 
                count += 1
            p = f.readline()
        f.close()		
        return p_dict 	

    def generate_postag_dict(self):#测试完毕 
        f = open(self.postag_path, encoding='UTF-8')
        pos = f.readline().strip('\n')
        postag_dict = {}
        count = 1
        while pos:
            postag_dict[pos] = count 
            count += 1 
            pos = f.readline().strip('\n')
        f.close()
        return postag_dict

    def generate_maxlen_and_worddict(self):#测试完毕 
        max_len, index = 0, 1
        word_dict = {}
        data_frame = pd.concat([self.train_data, self.test_data])
        for i, rows in tqdm(data_frame.iterrows()):
            max_len = max(max_len, len(rows['postag']))
            for ele in rows['postag']:
                if ele['word'] not in word_dict:
                    word_dict[ele['word']] = index 
                    index += 1
        return max_len, word_dict 

    def generate_postag_embedding(self):
        return np.float32(np.random.uniform(-0.1, 0.1, size = [25, 14]))

    def generate_word_embedding(self):#测试完毕 
        def convert_to_dict(path):
            f = open(path, encoding='UTF-8')
            data = f.readlines()
            _dict = {}
            for ele in tqdm(data):
                temp = ele.strip('\n').split('\t')
                _dict[temp[0]] = list(map(lambda x:float(x), temp[1].split(',')))
                f.close()
            return _dict 
		 #pre_word_embedding = convert_to_dict(self.pre_word_embedding_path)
		 #baike_word_embedding = convert_to_dict(self.baike_word_embedding_path)

        word_embedding = np.float32(np.random.uniform(-0.1, 0.1, size = [len(self.word_dict) + 1, 300]))#预训练维度是300 
        '''
		 for word, index in tqdm(self.word_dict.items()):
			 try:
				 word_embedding[index] = pre_word_embedding[word]
			 except:
				 try:
					 word_embedding[index] = baike_word_embedding[word]
				 except:
					 pass 
		 word_embedding[0, :] = np.zeros(300)
	    '''
        return word_embedding 

    def parse_word_embedding(self, batch_data):	#测试完毕
        word_embedding = np.zeros([len(batch_data), self.max_len])
        for i, (_, row) in enumerate(batch_data.iterrows()):
            for j, ele in enumerate(row['postag']):
                if ele['word'] in self.word_dict:
                    word_embedding[i][j] = self.word_dict[ele['word']]			
        return word_embedding 

    def parse_pos_embedding(self, batch_data):
        pos_embedding = np.zeros([len(batch_data), 2 * self.max_len])
        for i, (_, row) in enumerate(batch_data.iterrows()):
            for j, ele in enumerate(row['postag']):
                if j >= 0 and j <= len(row['postag']) - 1:
                    pos_embedding[i][j] = j + 1 
                    pos_embedding[i][j + len(row['postag'])] = len(row['postag']) - j
        return pos_embedding  		

    def parse_postag(self, batch_data):	#测试完毕
        postag = np.zeros([len(batch_data), self.max_len])
        for i, (_, row) in enumerate(batch_data.iterrows()):
            for j, ele in enumerate(row['postag']):
                postag[i][j] = self.postag_dict[ele['pos']]
        return postag

    def parse_p_labels(self, batch_data):	#测试完毕
        p_labels = np.zeros([len(batch_data), len(self.p_dict)])
        for i, (_, row) in enumerate(batch_data.iterrows()):
            for j, 	ele in enumerate(row['spo_list']):
                p_labels[i][self.p_dict[ele['predicate']]] = 1
        return p_labels	

    def parse_features(self, batch_data, features):  #测试完毕
        batch_features = {}
        for feature in features:
            batch_features[feature] = self.feature_func_dict[feature](batch_data)
        labels = self.parse_p_labels(batch_data)
        return batch_features, labels 	 	

    def generate_p_batch(self, batch_size, data, features = ['word_embedding', 'postag']): #测试完毕
        nums = len(data)
        index = 0
        while index < nums:
            batch_data = data.iloc[index:index + batch_size, :]
            batch_features, labels = self.parse_features(batch_data, features)
            yield batch_features, labels  
            index = index + batch_size 			
   
    def generate_ps_batch(self):
        pass
    
    def generate_pso_batch(self):
        pass

if __name__ == '__main__':
	train_data_path_list = ['../../data/train_data.json']
	test_data_path = '../../data/dev_data.json'
	pre_word_embedding_path = '../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
	baike_word_embedding_path = '../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table'
	postag_path = '../../data/pos'
	p_path = '../../data/all_50_schemas'
	process_data = process_data(train_data_path_list, test_data_path, pre_word_embedding_path, baike_word_embedding_path, postag_path, p_path)
	print('p_dict:{}'.format(process_data.p_dict))
	print('postag_dict:{},word_dict:{}'.format(len(process_data.postag_dict), len(process_data.word_dict)))
	train_data = process_data.generate_p_batch(256, process_data.train_data)
	print(train_data.__next__())
