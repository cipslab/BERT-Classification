import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from utils import get_split
from transformers import BertTokenizer

class DisinfoDataset(Dataset):
    def __init__(self, args):
        self.mode = mode
        # train['text_split'] = train['raw_text'].apply(get_split)
        test['text_split'] = test['raw_text'].apply(get_split, args=(400,50))
        # self.train = self.split_text(train)
        self.test = self.split_text(test)
        self.test_story_ids = self.test["story_id"].unique()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    
    
    def split_text(self, df):
        tmp = []
        for i in tqdm(range(len(df))):
            for j in range(len(df.iloc[i].text_split)):
        #         chunk_num = str(train.iloc[i]['story_id']) + '_' + str(j)
                chunk_num = j
                tmp.append(
                {'story_id': df.iloc[i]['story_id'],
                    'chunk_num': chunk_num,
                    'raw_text': df.iloc[i]['raw_text'],
                    'text_chunk': df.iloc[i]['text_split'][j],
                    'label': -1 if 'label' not in df else df.iloc[i]['label'] }
                )
        df = pd.DataFrame(tmp) 
        return df
    
    def tokenize(self,text):
        tokens = self.tokenizer.encode(
            text,                          #sentence to encode.
            add_special_tokens = True,   # Add '[CLS]' and '[SEP]'
            max_length = 512,  
            truncation=True,
            padding = 'max_length'          # Truncate all the sentences.
    #             return_tensors = 'pt'        # Return pytorch tensors.
        )

        return tokens

    def __getitem__(self, idx):
        if self.mode == "train":
            row = self.train.iloc[idx]
            story_id = row["story_id"]
            chunk_num = row["chunk_num"]
            tokens = torch.tensor(self.tokenize(row["text_chunk"]))
            label = row["label"]
            att_mask = torch.tensor([int(token_id > 0) for token_id in tokens])

            return story_id, chunk_num, tokens, att_mask, label
        else:
            story_id = self.test_story_ids[idx]
            label = self.test[self.test["story_id"]==story_id]["label"].values[0]
            token_list = []
            attn_mask = []
            for text_chunk in self.test[self.test["story_id"] == story_id]["text_chunk"].values:
                tokens = self.tokenize(text_chunk)
                token_list.append(tokens)
                attn_mask.append([int(token_id > 0) for token_id in tokens])
            # print(token_list, attn_mask)
            return story_id, torch.tensor(token_list), torch.tensor(attn_mask), label

    def __len__(self):
        return len(self.train) if self.mode=="train" else len(self.test_story_ids)


def collate_fn(batch):
    split_lens = [x.shape[0] for _,x,_,_ in batch]
    max_split_len = max(split_lens)
    batch_size = len(batch)
    token_tensor = torch.zeros(len(batch), max_split_len, 512)
    attn_mask_tensor = torch.zeros(len(batch), max_split_len, 512)
    story_ids = []
    labels = []
    for i, (story_id, token_list, attn_mask, label) in enumerate(batch):
        story_ids.append(story_id)
        labels.append(label)
        token_tensor[i][:split_lens[i]] = token_list
        attn_mask_tensor[i][:split_lens[i]] = attn_mask

    token_tensor = token_tensor.type('torch.LongTensor')
    attn_mask_tensor = attn_mask_tensor.type('torch.FloatTensor')
    return torch.tensor(story_ids), token_tensor, attn_mask_tensor, torch.tensor(labels), torch.tensor(split_lens)


