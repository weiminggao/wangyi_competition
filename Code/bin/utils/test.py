#import pandas as pd 
#
#<<<<<<< HEAD
#f = open('../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5')
#f1 = open('../../data/embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table', 'w')
#=======
#f = open('F://wangyi_competition-master//Code//data//embedding//sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5', encoding='UTF-8')
#f1 = open('F://wangyi_competition-master//Code//data//embedding//sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.table', 'w', encoding='UTF-8')
#>>>>>>> 'cnn_p'
#line = f.readline()
#line = f.readline()
#count = 0
#while line:
#	line = list(line.strip(' \n').split(' '))
#	count += 1
#	data = line[0] + '\t' + ','.join(line[1:]) + '\n'
#	f1.write(data)
#	line = f.readline()
#	if count  % 2000 == 0:
#		print('count:{}'.format(count))

#'''
#def df_convert_dict(df):
#			_dict = {}
#			for i, row in tqdm(df.iterrows()):
#				_dict[row[0]] = list(map(lambda x: float(x), row[1].split(',')))
#			return _dict 
#		pre_word_embedding = df_convert_dict(pd.read_table(self.pre_word_embedding_path, delimiter = '\t', header = None, quoting = csv.QUOTE_NONE, error_bad_lines = False))
#		baike_word_embedding = df_convert_dict(pd.read_table(self.baike_word_embedding_path, delimiter = '\t', header = None, quoting = csv.QUOTE_NONE, error_bad_lines = False)) 
#'''
#'''
#f = open('test')
#data = f.readlines()
#for ele in data:
#	print(ele.strip('\n'))
#'''
#import json
#read_f = open('../../data/dev_data_pso.json', encoding = 'UTF-8')        
#write_f = open('../../data/dev_data_pso_distinct.json', 'w', encoding = 'UTF-8')
#line = read_f.readline()
#texts = []
#while line:
#    data = json.loads(line)
#    if data['text'] not in texts:
#        texts.append(data['text'])
#        write_f.write(json.dumps(data, ensure_ascii = False) + '\n')
#    line = read_f.readline()
#import pandas as pd
#data = pd.read_json('../../data/dev_data_pso.json', lines = True, encoding="utf8")
#print(len(data))
#print('kkkkk')
read_f = open('../../data/test_data_postag.json', encoding='UTF-8')  #将文件分成6份
data = read_f.readlines()
num = len(data)
base = int(num / 6)
#for i in range(6):
write_f =  open('../../data/test_data_postag' + str(1)+ '.json', 'w', encoding='UTF-8')
write_f.writelines(data[65868:num])
write_f.close()
read_f.close()








