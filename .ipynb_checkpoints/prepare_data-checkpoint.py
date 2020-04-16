import re
import json
import pickle
import pandas as po
from tqdm import tqdm_notebook

with open('data/squad/train-v2.0.json', 'r') as train_file:
    train_data = json.load(train_file)

data = []

for i, title in tqdm_notebook(enumerate(train_data['data'])):
    for paragraph in train_data['data'][i]['paragraphs']: 
        #print(paragraph['context'])
        for qa_pair in paragraph['qas']:
            #print(qa_pair['question'])
            for ans in qa_pair['answers']:
                item = {}
                item['context'] = paragraph['context']
                item['question'] = qa_pair['question']
                item['answer'] = ans['text']
                #print(ans['text'])
                data.append(item)

data[43]

data[44]

with open('data/squad/data.pkl', 'wb') as file:
    pickle.dump(data, file)

# +
with open('data/squad/data.pkl', 'rb') as file:
    data = pickle.load(file)

data[64]

X = []
data_cleaned = []
for c_a_q in tqdm_notebook(data):
    context = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in c_a_q['context'].split(' ')])
    question = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in c_a_q['question'].split(' ')])
    answer = ' '.join([re.sub(r'[\W_]', '', w.lower()) for w in c_a_q['answer'].split(' ')])
    
    x = tokenizer.cls_token + ' ' + context + ' ' + tokenizer.sep_token + ' ' + answer + ' ' + tokenizer.sep_token + ' ' + tokenizer.mask_token 
    X.append(x)
    
    item = {}
    item['context'] = context
    item['question'] = question
    item['answer'] = answer
    data_cleaned.append(item)

with open('data/squad/data_cleaned.pkl', 'wb') as file:
    pickle.dump(data_cleaned, file)


# -











# print(train_data['data'][i]['paragraphs'][0]['context'])
#         for question in train_data['data'][i]['paragraphs'][0]['qas']:
#             print('##')
#             print(question['question'])
#             for answer in question['answers']:
#                 print(answer['text'])
#         print()
# print(train_data['data'][i]['paragraphs'][0]['context'])
#     for question in train_data['data'][i]['paragraphs'][0]['qas']:
#         print('##')
#         print(question['question'])
#         for answer in question['answers']:
#             print(answer['text'])
#     print()

# questions = []
# for i, title in enumerate(train_data['data']):
#     for paragraph in train_data['data'][i]['paragraphs']:
#         for qa_pair in paragraph['qas']:
#             questions.append(qa_pair['question'])

# with open('data/squad/train-questions-only.pkl', 'wb') as train_questions_file:
#     pickle.dump(questions, train_questions_file)







# def load_rrc():
#     with open('data/rrc/rest/train.json', 'r') as train_file:
#         train_data = json.load(train_file)
#
#     train_df = po.DataFrame(columns = ['Paragraph', 'Question', 'Answer'])
#     for review in train_data['data']:
#         for paragraph in review['paragraphs']:
#             for qas in paragraph['qas']:
#                 row = {'Paragraph': paragraph['context'],
#                        'Question': qas['question'],
#                        'Answer': qas['answers'][0]['text']
#                       }
#                 train_df = train_df.append(row, ignore_index=True)
#
#     train_df
#
#     train_df.to_csv('data/rrc/rest/train_df.csv', index = False)

# def load_squad_questions_only():
#     with open('data/squad/train-v2.0.json', 'r') as train_file:
#         train_data = json.load(train_file)
#     
#     questions = []
#     for i, title in enumerate(train_data['data']):
#         for paragraph in train_data['data'][i]['paragraphs']:
#             for qa_pair in paragraph['qas']:
#                 questions.append(qa_pair['question'])
#
#     with open('data/squad/train-questions-only.pkl', 'wb') as train_questions_file:
#         pickle.dump(questions, train_questions_file)

# load_squad_questions_only()

# with open('data/squad/train-questions-only.pkl', 'rb') as train_questions_file:
#     questions = pickle.load(train_questions_file)


