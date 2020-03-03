#  -*- coding: utf-8 -*-

import json
import numpy as np
from scipy import stats
import pandas as pd 

import numpy as np
from collections import Counter
def auc(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels) 
    c = Counter(labels)
    pos_cnt, neg_cnt = c[1], c[0]

    t = sorted(zip(probs, labels), key=lambda x: (x[0], x[1]), reverse=True)
    pos_than_neg, find_neg = 0, 0
    tmp_set = set()

    for i in range(len(t)):
        """
        处理多个样本的预测概率值相同的情况
        """
        if i + 1 < len(t) and t[i][0] == t[i + 1][0]:
            tmp_set.add(i)
            tmp_set.add(i + 1)
            continue

        if len(tmp_set) > 1:
            c = Counter([t[i][1] for i in tmp_set])
            pos, neg = c[1], c[0]
            find_neg = find_neg + neg
            pos_than_neg += pos * (neg_cnt - find_neg + neg / 2)
            tmp_set.clear()
            continue

        if t[i][1] == 1:
            pos_than_neg += (neg_cnt - find_neg)
        else:
            find_neg += 1

    return 1.0 * pos_than_neg / (pos_cnt * neg_cnt)


def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels) 
    return (preds == labels).mean()

def pre_recall_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    preds1 = np.array(preds, dtype=np.float64)
    labels1 = np.array(labels, dtype=np.float64)
    print("-----")
    #print(type(preds1[0]))
    #s1, s2 = stats.normaltest(labels1, axis = None)
    #print(s1,s2)
    #for i in preds1:
    #    if(i==np.nan):
    #        print (i)
    #print(preds1[0])
    s3, s4 = stats.normaltest(preds1, axis = None)
    print(s3,s4)
    #print("xiaoxiaode")
    #print(preds)
    data = pd.DataFrame({'labels':labels1,'preds':preds1})
    print(data.corr('spearman'))
    print("-----")
    # recall=TP/(TP+FN)
    
    tp = np.sum((labels == '1') & (preds == '1'))
    tn = np.sum((labels == '0') & (preds == '0'))
    fp = np.sum((labels == '0') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    r = tp * 1.0 / (tp + fn)
    # Precision=TP/(TP+FP)
    p = tp * 1.0 / (tp + fp)
    epsilon = 1e-31
    f1 = 2 * p * r / (p+r+epsilon)
    p0 = (tp+tn) *1.0 / len(preds)
    label_1 = np.sum(labels == '1')
    label_0 = np.sum(labels == '0')
    pred_0 = np.sum(preds =='0')
    pred_1 = np.sum(preds=='1')
    pe = ((label_1 * pred_1 + label_0 * pred_0) *1.0) / (len(preds)*len(preds))
    k = (p0-pe) * 1.0 / (1-pe)
    print('p0:{}, pe:{}, kappa:{}'.format(p0,pe,k))
    return p, r, f1


def res_evaluate(res_dir="./outputs/predict/predictions.json", eval_phase='test'):
    if eval_phase == 'test':
        data_dir="./0222_ready/part-00000-train"
    elif eval_phase == 'dev':
        data_dir="./data/dev.tsv"
    else:
        assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'
    cc = 0    
    labels = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            line = line.split("\t")
            label = line[2][:-1]
            label = str(label)
            
            if label=='2' or label == '3':
                label = '1'
            if label=='label':
                continue
            if label=='0':
                cc +=1
            labels.append(label)
    file.close()
    print(cc)

    preds = []
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['cos_sim'][0]
            preds.append(str(pred))
    file.close()
    '''
    with open('label00.tsv','w+') as f0:
        with open('label11.tsv','w+') as f1:
            with open('label22.tsv','w+') as f2:
        
    for i in range(len(labels)):
                    if(labels[i]=='0'):
                        f0.write(preds[i]+'\n')
                    if(labels[i]=='1'):
                        f1.write(preds[i]+'\n')
                    if(labels[i]=='2'):
                        f2.write(preds[i]+'\n')
    '''
    p = []
    l = []
    for i in range(len(labels)):
        if preds[i]!='-inf' and preds[i]!='inf' and preds[i]!='nan' :
            p.append(preds[i])
            l.append(labels[i])
    #        print('{}:{}'.format(i,preds[i]))
     #   else:
    #        print('{}:{}'.format(i,preds[i]))
    assert len(labels) == len(preds), "prediction result({}) doesn't match to labels({})".format(len(p),len(l))
    print('data num: {}'.format(len(l)))
    # p, r, f1 = pre_recall_f1(p, l)
    a = auc(p, l)
    print(a)
    #accuracy(p,l)
#    print("accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy(p, l), p, r, f1))

res_evaluate()
