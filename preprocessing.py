from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
import pickle
from xgboost import XGBClassifier

def model_extract(input_string):
    param ={'maxLen' :256,}
    # model = AutoModel.from_pretrained("ai4bharat/indic-bert")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

    def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
        padded_sequences = []
        for seq in sequences:
            if padding == 'pre':
                padded_seq = np.pad(seq, (maxlen - len(seq), 0), 'constant', constant_values=value)
            elif padding == 'post':
                padded_seq = np.pad(seq, (0, maxlen - len(seq)), 'constant', constant_values=value)
            else:
                raise ValueError("Padding should be 'pre' or 'post'.")

            if truncating == 'pre':
                padded_seq = padded_seq[-maxlen:]
            elif truncating == 'post':
                padded_seq = padded_seq[:maxlen]
            else:
                raise ValueError("Truncating should be 'pre' or 'post'.")

            padded_sequences.append(padded_seq)

        return np.array(padded_sequences, dtype=dtype)


    def create_attention_masks(input_ids):
        attention_masks = []
        for seq in tqdm(input_ids):
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return np.array(attention_masks)

    def getFeaturesandLabel(single_string, label):
        # Wrap the single string in a list
        sentences = ["[CLS] " + single_string + " [SEP]"]

        # Tokenize and preprocess
        tokenizer_texts = list(map(lambda t: tokenizer.tokenize(t)[:512], tqdm(sentences)))
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenizer_texts)]

        # Pad sequences and create attention masks
        input_ids = pad_sequences(sequences=input_ids, maxlen=param['maxLen'], dtype='long', padding='post', truncating='post')
        attention_masks_data = create_attention_masks(input_ids)

        # Convert to torch tensors
        X_data = torch.tensor(input_ids)
        attention_masks_data = torch.tensor(attention_masks_data)
        y_data = torch.tensor(label)

        return X_data, attention_masks_data, y_data 
    
    text_input=input_string
    label_input = [0]
    X_data, attention_masks_data, y_data = getFeaturesandLabel(text_input, label_input)
    return X_data

match=["సచ్చినోడ","పప్పు నాయుడు","నీచుడు","యెడవా","పనికిరాణి వాడు","దున్నపోతు","పిచ్చి","దరిద్రుడు","దొంగ","దోచేసాడు","సైకో","లపాకి","కొజ్జ","ముండ","ఎదవ","అడుక్కుతిను","దద్దమ్మ","సిగ్గులేదా","ఎర్రిపుకు","సన్నాసి","పోరంబోకు"]

input_string = "అక్కడ ఏమి వుంది అని అంతా ఓవర్ యాక్షన్"
ans=model_extract(input_string)
print('Torch.tensor vaiable: ',ans)
