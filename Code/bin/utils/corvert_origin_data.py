# -*- coding: utf-8 -*-
import json
import copy

def find_word_position(postag, s):
    index = []
    for start, (word, pos) in enumerate(postag):
        if word == s[0: len(word)]:
            s_index = len(word)
            pos_index = start + 1
            while s_index < len(postag) and pos_index < len(postag):
                if postag[pos_index] == s[s_index: s_index + len(postag[pos_index])]:
                    s_index = s_index + len(postag[pos_index])
                    pos_index = pos_index + 1
                else:
                    break
        if s_index == len(postag):
            index.append(start)
            index.append(pos_index)
            return index
    return False
    
def convert_to_ps(path):
    read_f = open(path[0])        
    write_f = open(path[1], 'w')
    line = read_f.readline()
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
                ps[spo['predicate']].append(spo['subject'])
            
        for p in ps:
            _data = copy.deepcopy(data)
            _data['p'] = p
            _data['s_index'] = []
            for s in ps[p]:
                s_index = find_word_position(data['postag'], s)
                if s_index == False:
                    print('s的分词结果与句子的分词有差异:{}'.format(data))
                    continue
                else:
                    _data['s_index'].append(s_index)
            write_f.write(json.dumps(_data))
    read_f.close()
    write_f.close()
        
def convert_to_pso(path):
    read_f = open(path[0])
    write_f = open(path[1], 'w')
    line = read_f.readline()
    while line:
        data = json.loads(line)
        if 'spo_list' not in data:
            print('不存在关系:{}'.format(data))
            continue
        
        for spo in data['spo_list']:
            _data = copy.deepcopy(data)
            _data['p'] = spo['predicate']
            _data['s'] = spo['subject']
            _data['o_index'] = []
            o_index = find_word_position(data['postag'], spo['object'])
            if o_index == False:
                print('o的分词结果与句子的分词有差异:{}'.format(data))
                continue
            else:
                _data['o_index'].append(o_index)
            write_f.write(json.dumps(_data))        
    read_f.close()
    write_f.close()
    
if __name__ == '__main__':
    ps_paths = [['../../data/train_data.json', '../../data/train_data_ps.json'], ['../../data/dev_data.json', '../../data/dev_data_ps.json']]
    pso_paths = [['../../data/train_data.json', '../../data/train_data_pso.json'], ['../../data/dev_data.json', '../../data/dev_data_pso.json']]
    for i in range(len(ps_paths)):
        convert_to_ps(ps_paths[i])
        convert_to_pso(ps_paths[i])