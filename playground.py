# +
input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

last_hidden_states.shape

input_ids = torch.tensor(tokenizer.encode(X[457], add_special_tokens=False))

input_ids.shape

tokenizer.mask_token_id

input_ids[-1]

tokenizer.vocab_size

#start with a per word loss model then try something with rollout

'data/squad/data_cleaned.pkl'

#data_cleaned[3124]

with open('data/squad/data_cleaned.pkl', 'rb') as file:
    data = pickle.load(file)

# +
torch.tensor(tokenizer.encode(word, add_special_tokens=False))
x
out.shape

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
#need to make a custom dataloader with a step method
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

