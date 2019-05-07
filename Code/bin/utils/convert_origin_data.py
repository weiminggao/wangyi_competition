# -*- coding: utf-8 -*-
import json
import copy

def find_sentence_position(postag, s): 
    index = []
    for start, word_pos in enumerate(postag):
        s_index = 0
        word = word_pos['word']
        if word == s[0: len(word)]:
            s_index = len(word)
            word_pos_index = start + 1
            while s_index < len(s) and word_pos_index < len(postag):
                if postag[word_pos_index]['word'] == s[s_index: s_index + len(postag[word_pos_index]['word'])]:
                    s_index = s_index + len(postag[word_pos_index]['word'])
                    word_pos_index = word_pos_index + 1
                else:
                    break
        if s_index == len(s):
            index.append(start)
            index.append(word_pos_index)
            return index
    return False
    
def convert_to_ps(path): #测试完毕
    read_f = open(path[0], encoding = 'UTF-8')        
    write_f = open(path[1], 'w', encoding = 'UTF-8')
    error_write_f = open(path[2], 'w', encoding = 'UTF-8')
    line = read_f.readline()
    count = 1
    error_count = 1
    while line:
        data = json.loads(line)
        if 'spo_list' not in data:
            print('不存在关系:{}'.format(data))
            continue
        ps = {}
        for spo in data['spo_list']:
            if spo['predicate'] not in ps:
                ps[spo['predicate']] = [spo['subject']]
            else:
                if spo['subject'] not in ps[spo['predicate']]:
                    ps[spo['predicate']].append(spo['subject'])

        for p in ps:
            _data = copy.deepcopy(data)
            _data['p'] = p
            _data['s_index'] = []
            for s in ps[p]:
                s_index = find_sentence_position(data['postag'], s)
                if s_index == False:
                    #print('s的分词结果与句子的分词有差异')
                    error_write_f.write(json.dumps(_data, ensure_ascii = False) + '\n')
                    error_count += 1
                    continue
                else:
                    _data['s_index'].append(s_index)
            if len(_data['s_index']) != 0: 
                write_f.write(json.dumps(_data, ensure_ascii = False) + '\n')
        
        line = read_f.readline()
        if count % 10000 == 0:
            print('第{}条'.format(count))
        count += 1
    print('总条目数:{},错误条目数:{}'.format(count, error_count))
    read_f.close()
    write_f.close()
    error_write_f.close()
        
def convert_to_pso(path):
    read_f = open(path[0], encoding = 'UTF-8')
    write_f = open(path[1], 'w', encoding = 'UTF-8')
    error_write_f = open(path[2], 'w', encoding = 'UTF-8')
    line = read_f.readline()
    count = 1
    error_count = 1
    while line:
        data = json.loads(line)
        if 'spo_list' not in data:
            print('不存在关系:{}'.format(data))
            continue
        
        for spo in data['spo_list']:
            _data = copy.deepcopy(data)
            _data['p'] = spo['predicate']
            
            s_index = find_sentence_position(data['postag'], spo['subject'])
            if s_index == False:
                #print('s的分词结果与句子的分词有差异')
                error_count += 1
                error_write_f.write(json.dumps(_data, ensure_ascii = False) + '\n')
                continue
            else:
                _data['s'] = []
                for i in range(s_index[0], s_index[1]):
                    _data['s'].append(data['postag'][i]['word'])
                    
            _data['o_index'] = []
            o_index = find_sentence_position(data['postag'], spo['object'])
            if o_index == False:
                #print('o的分词结果与句子的分词有差异')
                error_count += 1
                error_write_f.write(json.dumps(_data, ensure_ascii = False) + '\n')
                continue
            else:
                _data['o_index'].append(o_index)
            write_f.write(json.dumps(_data, ensure_ascii = False) + '\n')
        
        line = read_f.readline()
        if count % 10000 == 0:
            print('第{}条'.format(count))
        count += 1
    print('总条目数:{},错误条目数:{}'.format(count, error_count))
    read_f.close()
    write_f.close()
    error_write_f.close()
    
if __name__ == '__main__':
    ps_paths = [['../../data/train_data.json', '../../data/train_data_ps.json', '../../data/train_data_error_ps.json'], \
                ['../../data/dev_data.json', '../../data/dev_data_ps.json', '../../data/dev_data_error_ps.json']]
    pso_paths = [['../../data/train_data.json', '../../data/train_data_pso.json', '../../data/train_data_error_pso.json'], \
                 ['../../data/dev_data.json', '../../data/dev_data_pso.json', '../../data/dev_data_error_pso.json']]
    for i in range(len(ps_paths)):
#        convert_to_ps(ps_paths[i])
        convert_to_pso(pso_paths[i])