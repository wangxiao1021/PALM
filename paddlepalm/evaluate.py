#  -*- coding: utf-8 -*-

import json
import numpy as np
from collections import Counter
def auc(probs, labels):
    probs = np.array(probs, dtype='float32')
    labels = np.array(labels, dtype='float32')
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
            pos_than_neg += pos * (neg_cnt - find_neg + neg / 2 *1.0)
            tmp_set.clear()
            continue

        if t[i][1] == 1:
            pos_than_neg += (neg_cnt - find_neg)
        else:
            find_neg += 1
    epsilon = 1e-31
    return 1.0 * pos_than_neg / (pos_cnt * neg_cnt + epsilon) 
def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels) 
    return (preds == labels).mean()

def pre_recall_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    # recall=TP/(TP+FN)
    tp = np.sum((labels == '1') & (preds == '1'))
    fp = np.sum((labels == '0') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    r = tp * 1.0 / (tp + fn)
    # Precision=TP/(TP+FP)
    p = tp * 1.0 / (tp + fp)
    epsilon = 1e-31
    f1 = 2 * p * r / (p+r+epsilon)
    return p, r, f1


def do_eval(step):
    # if eval_phase == 'test':
    #     data_dir="./data/test.tsv"
    # elif eval_phase == 'dev':
    #     data_dir="./data/dev.tsv"
    # else:
    #     assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'
    res_dir="./outputs/predict/pred-"+str(step) 
    labels = []
    # with open(data_dir, "r") as file:
    #     first_flag = True
    #     for line in file:
    #         line = line.split("\t")
    #         label = line[2][:-1]
    #         if label=='label':
    #             continue
    #         labels.append(str(label))
    # file.close()

    preds = []
    probs = [] 
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            prob = line['probs'][1]
            label = line['ori-label']
            preds.append(str(pred))
            labels.append(str(label))
            probs.append(str(prob))
    file.close()
    assert len(labels) == len(preds), "prediction result({}) doesn't match to labels({})".format(len(preds),len(labels))
    # print('data num: {}'.format(len(labels)))
    p, r, f1 = pre_recall_f1(preds, labels)
    ac = accuracy(preds, labels)
    au = auc(probs,labels)
    return au, ac, p, r, f1, len(labels)
    # print("accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy(preds, labels), p, r, f1))

# res_evaluate()
