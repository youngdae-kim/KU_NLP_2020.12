from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from tqdm.notebook import tqdm as tqdm_nb
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json



#####################################
# 0. hypterpameter Setting
# - 학습에 이용되는 하이퍼파라미터 세팅
# - pretrained BERT 모델 세팅
#####################################
print("0. hypterpameter Setting")
learning_rate = 1e-5
n_epoch = 8

pretrained_weights = 'bert-base-uncased'
# pretrained_weights = 'bert-large-cased-whole-word-masking'


#####################################
# 1. json file to train&dev data Setting
# - Train & dev json 파일에서 학습 가능한 데이터 Set 으로 변환
#####################################
print("1. data Set")

data = {'train': {'speaker': [], 'utterance': [], 'emotion': []},
        'dev': {'speaker': [], 'utterance': [], 'emotion': []},
        'test': {'speaker': [], 'utterance': [], 'emotion': []}}

file_path = 'data_ENG/Friends/'

for dtype in ['train', 'dev', 'test']:
  for dialog in json.loads(open(file_path+'friends_' + dtype + '.json').read()):
      for line in dialog:
          data[dtype]['speaker'].append(line['speaker'])
          data[dtype]['utterance'].append(line['utterance'])
          data[dtype]['emotion'].append(line['emotion'])

  e2i_dict = dict((emo, i) for i, emo in enumerate(set(data['train']['emotion'])))
  i2e_dict = {i: e for e, i in e2i_dict.items()}


#####################################
# 2. model & Layer Setting
# - pretrained BERT 를 활용하여 tokenizer, model 세팅
# - 문장을 분석을 위한 단위(Token)로 쪼개고 BERT 입력 변수로 변환
# - utterance -> tokens -> ids -> input_tensor -> hidden_tensor -> logit
#####################################
print("2. model & Layer Setting")

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    self.bert_model = BertModel.from_pretrained(pretrained_weights)
    self.linear = torch.nn.Linear(768, len(e2i_dict))

  def forward(self, utterance):
    tokens = self.bert_tokenizer.tokenize(utterance)
    tokens = ['[CLS]'] + tokens + ['[SEP]'] # (len)
    ids = [tokenizer.convert_tokens_to_ids(tokens)] # (bat=1, len)
    input_tensor = torch.tensor(ids).cuda()

    hidden_tensor = self.bert_model(input_tensor)[0] # (bat, len, hid)
    hidden_tensor = hidden_tensor[:, 0, :] # (bat, hid)
    logit = self.linear(hidden_tensor)
    return logit


#####################################
# 3. f1-evaluate 
# - 예측 vs label 데이터에 대한 F1-score 평가
#####################################
print("3. f1-evaluate ")

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(true_list, pred_list):
  precision = precision_score(true_list, pred_list, average=None)
  recall = recall_score(true_list, pred_list, average=None)
  micro_f1 = f1_score(true_list, pred_list, average='micro')
  print('precision:\t', ['%.4f' % v for v in precision])
  print('recall:\t\t', ['%.4f' % v for v in recall])
  print('micro_f1: %.6f' % micro_f1)


#####################################
# 4. traning
# - loss fuction(adam, adamax 등) 및 hyperparameter 변경을 통한 학습 작업 실행
#####################################
print("4. traning")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = Model()
model.cuda()
criterion = torch.nn.CrossEntropyLoss()  # LogSoftmax & NLLLoss

# optimizer = torch.optim.Adam(model.parameters(), learning_rate)
optimizer = torch.optim.Adamax(model.parameters(),learning_rate)

for i_epoch in range(n_epoch):
    print('i_epoch:', i_epoch)

    model.train()
    for i_batch in tqdm_nb(range(len(data['train']['utterance']))):
        #if i_batch%100==0 : print("{}번째 작업 / 총 {}건".format(i_batch, len(data['train']['utterance'])))
        logit = model(data['train']['utterance'][i_batch])
        target = torch.tensor([e2i_dict[data['train']['emotion'][i_batch]]]).cuda()
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    pred_list, true_list = [], []
    for i_batch in tqdm_nb(range(len(data['dev']['utterance']))):
        #if i_batch % 100 == 0: print("{}번째 작업 / 총 {}건".format(i_batch, len(data['dev']['utterance'])))
        logit = model(data['dev']['utterance'][i_batch])
        _, max_idx = torch.max(logit, dim=-1)
        pred_list += max_idx.tolist()
        true_list += [e2i_dict[data['dev']['emotion'][i_batch]]]
    evaluate(pred_list, true_list)  # print results


#####################################
# 5. test
#####################################

print(i2e_dict[torch.max(model("Alright, whadyou do with him?"), dim=-1)[1].tolist()[0]])
