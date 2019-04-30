# **BaiduNLP_Baseline:**
	该模型的的pipeline包含两个模型：
	* p模型，用于识别p也就是关系。其输入是word embedding和pos embedding，输出是关系（可能输出多个关系）
	* so模型，用于识别so,也就是subject和object，其输入是word embedding和pos embedding和p关系，输出是BIOE，这里跟分词的意思差不多。
	* 跑了37轮，precision:0.642313614353, recall:0.657935038515, f1:0.650030487116
# **模型的思考与用注意力模型的思考：**
	* baseline的正确与否依赖于p模型，但好处是可以将多个单词组合成正确的实，也就是so模型的输出是BIEO模型。
	* 假如利用att的话，输出是关系。需要先处理好实体，方法１）可以自己先分好实体（但是这样的话依赖与分好的实体的正确性），好像目前只有这种可能了，另外当输出的概率大于多少时会选择这个关系是个问题。
