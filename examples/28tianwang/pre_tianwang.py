 # coding=utf-8
import paddlepalm as palm
import json
from paddlepalm.distribute import gpu_dev_count
from glob import glob
from paddlepalm.evaluate import do_eval


def eval(pred_model_path, predict_file):
      ss_path='outcome'
      # configs
      max_seqlen = 256
      batch_size = 8
      vocab_path = './pretrain/ernie-zh-base/vocab.txt'
      # predict_file = './data/test.tsv'
      random_seed = 1
      config = json.load(open('./pretrain/ernie-zh-base/ernie_config_4L.json'))
      input_dim = config['hidden_size']
      num_classes = 2
      task_name = 'tianwang-4l'
      pred_output = './outputs/predict/'
      print_steps = 20
      # pre_params = './pretrain/ernie-zh-base/params'

      # -----------------------  for prediction ----------------------- 

      # step 1-1: create readers for prediction
      # print('prepare to predict...')
      predict_match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
      # step 1-2: load the training data
      predict_match_reader.load_data(predict_file, batch_size)

      # step 2: create a backbone of the model to extract text features
      pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

      # step 3: register the backbone in reader
      predict_match_reader.register_with(pred_ernie)

      # step 4: create the task output head
      match_pred_head = palm.head.Match(num_classes, input_dim, phase='predict')

      # step 5-1: create a task trainer
      trainer = palm.Trainer(task_name)
      # step 5-2: build forward graph with backbone and task head
      trainer.build_predict_forward(pred_ernie, match_pred_head)

      # step 6: load checkpoint
      trainer.load_ckpt(pred_model_path)

      # step 7: fit prepared reader and data
      trainer.fit_reader(predict_match_reader, phase='predict')

      # step 8: predict
      # print('predicting..')
      trainer.predict(print_steps=print_steps, output_dir=pred_output, tt_step=-1)
      with open(ss_path, 'a') as f:
            au, ac, p, r, f1, num = do_eval(-1, predict_file)
            f.write("file: {}\tstep: {}\tauc:{:.5f}\tacc:{:.5f}\tpre:{:.5f}\trecall:{:.5f}\tf1:{:.5f}\tnum:{}\n".format
                        (predict_file[6:][:-5], pred_model_path[19:], au, ac, p, r, f1, num))
      f.close()


if __name__ == '__main__':
      pred_model_path =  glob('./outputs/ckpt.step*')
      data = glob('check/*')

      for p in pred_model_path:
            for d in data:
                  eval(p, d)
