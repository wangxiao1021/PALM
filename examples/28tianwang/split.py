#coding=utf8
import sys
import os
import itertools
import random

pos_data = '0222_ready/part-00000-pos'
neg_data = '0222_ready/part-00000-neg'
train_data = '0222_ready/part-00000-train'
test_data = '0222_ready/part-00000-test'
# pos_data = 'xiao'
# neg_data = 'cas'

f = open(pos_data,'r')
f1 = open(neg_data,'r')
datapos = list(f.readlines())
random.shuffle(datapos)
dataneg = list(f1.readlines())
random.shuffle(dataneg)
# train_pos = datapos
# train_neg = dataneg
train_pos = datapos[:int(len(datapos) * 0.7)]
test_pos = datapos[int(len(datapos) * 0.7):]
train_neg = dataneg[:int(len(dataneg) * 0.7)]
test_neg = dataneg[int(len(dataneg) * 0.7):]
f.close()

with open(train_data,'w') as f2:
    with open(test_data, 'w') as f3:
        f2.write('text_a\ttext_b\tlabel\n')
        f3.write('text_a\ttext_b\tlabel\n')
        n = 0
        f = 0
        for i in train_pos:
            f2.write(i)
            n+=1
            if n%5==0 and f < len(train_neg):
                f2.write(train_neg[f])
                f+=1
        while f<len(train_neg):
            f2.write(train_neg[f])
            f+=1
        n = 0
        f = 0
        for i in test_pos:
            f3.write(i)
            n+=1
            if n%5==0 and f < len(test_neg):
                f3.write(test_neg[f])
                f+=1
        while f<len(test_neg):
            f3.write(test_neg[f])
            f+=1
        # for i,j in zip(test_pos, test_neg):
        #     f3.write(i+j)
            

f1.close()
f2.close()
f3.close()
print('train data and test_data saved in {} and {}.'.format(train_data, test_data))
