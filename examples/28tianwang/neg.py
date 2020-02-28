#coding=utf8
import sys
import os
import itertools
import random
from collections import Counter

in_data = '0222_process/part-00000'
pos_data = '0222_ready/part-00000-pos'
out_data = '0222_ready/part-00000-neg'
# in_data = 'ca.tsv'
# pos_data = 'xiao'
# out_data = 'cas'

bufb = []
buf = []
ff = open(in_data, 'r')
pairs = set()

ff.close()
pre=''
vv = []
# pairs={}
with open(in_data, 'r') as f:
    with open(out_data, 'w') as f1:
        # f1.write('text_a\ttext_b\tlabel\n')
        
        lwritten = 0
        for l in f.readlines():
            line = l.split('\t')
            line[1] = line[1][:-1]
            buf.append(line[0])
            bufb.append(line[1])
            pairs.add(frozenset([line[0],line[1]]))
 
        # exit()
        # print(pairs)
        # exit()
        ltotal = len(pairs)
        query = dict(Counter(buf))
     
        # exit()
        # text_b = set(frozenset(line.split('\t')[1]) for line in f)
        # print(text_b)
        # print(bufb)
        # words = list(word for word in itertools.chain.from_iterable(buf))
        random.shuffle(bufb)
        # print(bufb)
        # # print(words)
        # exit()
        

        for k, v in query.items():
            vv.append(v)
            n = 1 if v<8 else 2
            n += 1 if v>17 else 0
            while n>0:
                i = random.randint(0,len(bufb)-1)
                # print(i)
                pair=[k]
                pair+=[bufb[i]]
                # print(pair)
                if frozenset(pair) not in pairs:
                    f1.write('{}\t{}\t0\n'.format(k,bufb[i]))
                    lwritten+=1
                    n-=1
        d = dict(Counter(vv))
        print (d)
        s = 0
        for k, v in d.items():
            s+=v
        print(s)
        


        # for pair in itertools.izip(*[iter(words)] * 2):
        #     if frozenset(pair) not in pairs and lwritten != ltotal:
        #         f1.write('%s\t%s\t0\n' % pair)
        #         lwritten+=1

f.close()
f1.close()
with open(in_data,'r') as f:
    with open(pos_data, 'w') as f1:
        # f1.write('text_a\ttext_b\tlabel\n')
        for line in f.readlines():
            line = line.split('\t')
            line[1] = line[1][:-1]
            f1.write('{}\t{}\t1\n'.format(line[0],line[1]))
f.close()
f1.close()
print('total {} negtive samples and {} positive samples saved in {} and {}.'.format(lwritten,ltotal, out_data, pos_data))
print(lwritten*1.0/ltotal)