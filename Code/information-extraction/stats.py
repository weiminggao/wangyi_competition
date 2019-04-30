import pandas as pd
import os 
from tqdm import tqdm
import argparse
import json
import numpy as np  

def calculate_pre_rec_f1(sentences_real_num, sentences_predict_num, sentences_correct_num):
	precision = float(sentences_correct_num) / float(sentences_predict_num) 
	recall = float(sentences_correct_num) / float(sentences_real_num) 
	f1 = (2 * precision * recall) / (precision + recall)
	return precision, recall, f1 

def stats_p(origin_path, p_path):
	tqdm.pandas(desc = 'stats_p')
	origin_data = pd.read_json(origin_path, lines = True)
	p_data = pd.read_table(p_path, header = None)
	def convert(df):
		temp_df = pd.DataFrame({'text':[], 'p_list':[]})
		temp_df['p_list'] = [list(df.iloc[:, 1])]
		temp_df['text'] = json.loads(df.iloc[0, 0])['text']
		return temp_df 
	convert_data = p_data.groupby(by = 0).progress_apply(convert)
	temp_data = pd.merge(origin_data, convert_data, on = 'text').loc[:, ['spo_list', 'p_list']]

	sentences_real_num = sum(list(map(lambda x : len(x), list(origin_data.loc[:, 'spo_list']))))
	sentences_real_num1 = sum(list(map(lambda x : len(x), list(temp_data.loc[:, 'p_list']))))
	sentences_predict_num = sum(list(map(lambda x : len(x), list(convert_data.loc[:, 'p_list']))))
	sentences_correct_num = 0
	for index, row in tqdm(temp_data.iterrows()):
		p_list_res = sorted(row['p_list'])
		p_list_ori = sorted([ele['predicate'] for ele in row['spo_list']])#重复元素求交集
		index = -1
		for p_res in p_list_res:
			while index < len(p_list_ori) - 1:
				index += 1
				if p_res == p_list_ori[index]:
					sentences_correct_num += 1
					break					
	return calculate_pre_rec_f1(sentences_real_num, sentences_predict_num, sentences_correct_num)
	
def stats_f1(origin_path, result_path):
	origin_data = pd.read_json(origin_path, lines = True)
	result_data = pd.read_json(result_path, lines = True)
	sentences_real_num = sum(list(map(lambda x : len(x), list(origin_data.loc[:, 'spo_list']))))
	sentences_predict_num = sum(list(map(lambda x : len(x), list(result_data.loc[:, 'spo_list']))))
	temp_data = pd.merge(origin_data, result_data, on = 'text', suffixes = ['_ori', '_res']).loc[:, ['spo_list_ori', 'spo_list_res']]
	sentences_correct_num = 0	
	for index, row in tqdm(temp_data.iterrows()):
		spo_list_ori = row['spo_list_ori']	
		spo_list_res = row['spo_list_res']
		for spo in spo_list_res:
			if spo in spo_list_ori:
				sentences_correct_num += 1
	return calculate_pre_rec_f1(sentences_real_num, sentences_predict_num, sentences_correct_num)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--origin_path", type = str, default = 'dev_data.json')
	parser.add_argument("--result_path", type = str, default = 'dev_data.res')
	parser.add_argument("--p_path", type = str, default = 'dev_data.p')
	args = parser.parse_args()
	spo_precision, spo_recall, spo_f1 = stats_f1(os.path.join('/home/wmg/WorkSpace/Wangyi_Competition/Code/information-extraction/data', args.origin_path), os.path.join('/home/wmg/WorkSpace/Wangyi_Competition/Code/information-extraction/data', args.result_path))
	p_precision, p_recall, p_f1 = stats_p(os.path.join('/home/wmg/WorkSpace/Wangyi_Competition/Code/information-extraction/data', args.origin_path), os.path.join('/home/wmg/WorkSpace/Wangyi_Competition/Code/information-extraction/data', args.p_path))
	print('spo_precision:{}, spo_recall:{}, spo_f1:{}'.format(spo_precision, spo_recall, spo_f1))
	print('p_precision:{}, p_recall:{}, p_f1:{}'.format(p_precision, p_recall, p_f1))

