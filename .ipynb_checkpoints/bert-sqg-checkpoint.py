import re
import pickle
from tqdm import tqdm_notebook

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer

torch.set_default_tensor_type(torch.cuda.FloatTensor)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# model.save_pretrained('bert/')
# tokenizer.save_pretrained('bert/') 

# +
#tokenizer = BertTokenizer.from_pretrained('bert/')
#model = BertModel.from_pretrained('bert/')
# -

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
        tokens = torch.tensor([self.bert_tokenizer.encode(x, add_special_tokens=False)])
        last_hidden_states = model(tokens)[0]
        mask_token_h_state = last_hidden_states[:, -1, :]
    
        lin_out = self.linear(mask_token_h_state)    
        return lin_out


squad_data_path = 'data/squad/data_cleaned.pkl'
dataloader = SQuAD_dataloader(squad_data_path)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

bert_sqg = Bert_SQG(model, tokenizer) 

bert_sqg.cuda()

# +
#but we need to predict till the model outputs the [SEP] token
# -

target

bert_sqg.train()
for item in tqdm_notebook(dataloader):
    
    context = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['context'].split(' ')])
    question = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['question'].split(' ')])
    answer = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in item['answer'].split(' ')])
    output_question = ''
    
    x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + tokenizer.mask_token 
    
    for word in question.split(' '):
        target = torch.tensor(tokenizer.encode(word, add_special_tokens=False))
        out = bert_sqg(x)

        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        predicted_word = tokenizer.convert_ids_to_tokens([torch.argmax(torch.nn.Softmax(dim = 1)(out))])

        output_question = output_question + ' ' + predicted_word[0]
        print(output_question)
        print('##')
        x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + output_question + ' ' + tokenizer.mask_token 

    break

question

question.split(' ')

torch.tensor(tokenizer.encode(word, add_special_tokens=False))

x

out.shape

word = 'swimming'

torch.tensor([13246,65])[:1]

torch.tensor(tokenizer.encode(word, add_special_tokens=False))

output_question = ''
for word in question.split(' '):
    #print(word)
    target = torch.tensor(tokenizer.encode(word, add_special_tokens=False))
    out = bert_sqg(x)
    
    loss = criterion(out, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    predicted_word = tokenizer.convert_ids_to_tokens([torch.argmax(torch.nn.Softmax(dim = 1)(out))])
    
    output_question = output_question + ' ' + predicted_word[0]
    #print(predicted_word)
    print(output_question)
    print('##')
    x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + output_question + ' ' + tokenizer.mask_token 
    #print(x)


x

predicted_word[0]

tokenizer.encode('          yo          ma sup      ')

x

# +
#need to make a custom dataloader with a step method
# -

X[1]

data_cleaned[1]

a

for x in X:
    out = bert_sqg(x)
    loss = criterion()
    break

torch.argmax(out)

    with torch.no_grad():
        last_hidden_states = model(tokens)[0]
    print(last_hidden_states)
    print(last_hidden_states.squeeze()[-1, :])
    print(tokens.shape)
    break


