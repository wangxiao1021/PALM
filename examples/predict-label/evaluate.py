#  -*- coding: utf-8 -*-

import json
import numpy as np

def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels) 
    return (preds == labels).mean()

def pre_recall_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
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
        data_dir="./data/train.tsv"
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
            pred = line['label']
            preds.append(str(pred))
    file.close()
    assert len(labels) == len(preds), "prediction result({}) doesn't match to labels({})".format(len(preds),len(labels))
    print('data num: {}'.format(len(labels)))
    p, r, f1 = pre_recall_f1(preds, labels)
    print("accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy(preds, labels), p, r, f1))

res_evaluate()
