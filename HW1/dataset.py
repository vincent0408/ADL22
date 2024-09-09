from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent,
                           idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        token_list = [s['text'].split(' ') for s in samples]
        encoded_tokens = self.vocab.encode_batch(
            batch_tokens=token_list, to_len=self.max_len)

        try:
            s = samples[0]['intent']
            isTest = False
        except:
            isTest = True
        
        if (not isTest):
            intent_list = [self.label2idx(s['intent']) for s in samples]
            return (torch.LongTensor(encoded_tokens), torch.LongTensor(intent_list))
        else:
            return torch.LongTensor(encoded_tokens)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def collate_fn(self, samples):
        pad_to_len = 40
        token_list = [s['tokens'] for s in samples]
        
        encoded_tokens = self.vocab.encode_batch(
            batch_tokens=token_list, to_len=pad_to_len)
        try:
            s = samples[0]['tags']
            isTest = False
        except:
            isTest = True
        tk_len = [len(s['tokens']) for s in samples]
        
        if (not isTest):
            tags = []
            tag_collection = [s['tags'] for s in samples]
            for tag_list in tag_collection:
                t = []
                for tag in tag_list:
                    t.append(self.label2idx(tag))
                t.extend([-1] * (pad_to_len - len(t)))
                tags.append(t)
                
            return (torch.LongTensor(encoded_tokens), 
                    torch.LongTensor(tags), 
                    torch.LongTensor(tk_len))
        else:
            return (torch.LongTensor(encoded_tokens),
                    torch.LongTensor(tk_len))

