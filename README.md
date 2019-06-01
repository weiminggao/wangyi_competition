# Wangyi_competition
##总结：
###思路：利用级联模型进行句子的关系抽取（谓词、宾语、主语）
###1)相同的谓词可能有不同的主语--宾语对
###2)相同的谓词、相同的主语可能有不同的宾语
###考虑到以上条件：
###1)利用text-cnn的思路先分出谓词
###2)利用bi-lstm+谓词作为attention模型分出宾语
###3)利用bi-lstm+谓词作为attention+宾语作为attention模型（句子级别的embedding）分出主语
