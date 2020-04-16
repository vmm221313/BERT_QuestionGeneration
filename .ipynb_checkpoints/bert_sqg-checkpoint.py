# -*- coding: utf-8 -*-
# !pip install transformers

import re
import pickle
from tqdm import tqdm_notebook

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer

torch.set_default_tensor_type(torch.cuda.FloatTensor)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained('bert/') 
tokenizer.save_pretrained('bert/')

tokenizer = BertTokenizer.from_pretrained('bert/')
model = BertModel.from_pretrained('bert/')

class SQuAD_dataloader(Dataset):
    def __init__(self, pkl_file_path):
        
        self.pkl_file_path = pkl_file_path
        with open(self.pkl_file_path, 'rb') as file:
            self.data = pickle.load(file)

        self.idx = 0
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):
        if self.idx != len(self.data):
            self.idx += 1
            return self.data[self.idx - 1]
        else:
            self.idx = 0
            return self.data[self.idx]

class Bert_SQG(torch.nn.Module):
    def __init__(self, bert_model, bert_tokenizer):
        super(Bert_SQG, self).__init__()
        
        self.bert_model = bert_model #now training will be end-to-end
        self.bert_tokenizer = bert_tokenizer
        self.linear = torch.nn.Linear(768, bert_tokenizer.vocab_size)

    def forward(self, x):
        tokens = torch.tensor([self.bert_tokenizer.encode(x, add_special_tokens=False, max_length=510)])

        last_hidden_states = model(tokens)[0]
        mask_token_h_state = last_hidden_states[:, -1, :]
    
        lin_out = self.linear(mask_token_h_state)    
        return lin_out

squad_data_path = 'data/squad/data_cleaned.pkl'
dataloader = SQuAD_dataloader(squad_data_path)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

bert_sqg = Bert_SQG(model, tokenizer)

device = 'cuda'

bert_sqg.to(device)

# but we need to predict till the model outputs the [SEP] token

for epoch in range(10):
  bert_sqg.train()
  for item in tqdm_notebook(dataloader):
      
      context = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['context'].split(' ')])
      question = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['question'].split(' ')])
      answer = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['answer'].split(' ')])
      output_question = ''
      
      x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + tokenizer.mask_token 
      
      for word in question.split(' '):
          target = torch.tensor(tokenizer.encode(word, add_special_tokens=False))[:1]
          out = bert_sqg(x)

          try:
            loss = criterion(out, target)
          except ValueError:
            print(target)
            break
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
          optimizer.step()

          predicted_word = tokenizer.convert_ids_to_tokens([torch.argmax(torch.nn.Softmax(dim = 1)(out))])

          output_question = output_question + ' ' + predicted_word[0]
          #print(output_question)
          #print('##')
          x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + output_question + ' ' + tokenizer.mask_token 

      #print(output_question)
    
  torch.save(bert_sqg.state_dict(), 'saved_models/BERT_SQG_'+str(epoch+1)+'_epochs.pt')

