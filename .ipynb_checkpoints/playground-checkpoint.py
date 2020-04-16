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
