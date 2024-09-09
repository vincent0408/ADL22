from typing import Dict
import torch
from torch.nn import Embedding
from torch import nn
import torch.nn.functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(embeddings.size(1), 
                    hidden_size,
                    bidirectional=bidirectional,
                    dropout=dropout, 
                    num_layers=num_layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(2 * hidden_size + 300, num_class)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeddings = self.embed(batch)
        out = self.dropout(embeddings)
        out, _ = self.lstm(out) 
        out = torch.cat((out, embeddings), 2) 
        out = out.permute(0, 2, 1) 
        out = F.max_pool1d(out, out.size()[2]) 

        out = self.relu(out)
        out = self.dropout(out)
        out = self.out(out.squeeze(2))

        return out


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()       
        pad_tag = 40

        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.enc = Encoder(embeddings.size(1), hidden_size)
        self.dec = Decoder(hidden_size, pad_tag, num_class)
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeddings = self.embed(batch)
        out = self.enc(embeddings)
        out = self.dec(out)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size, 
                            num_layers=2,
                            bidirectional= True, 
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x 


class Decoder(nn.Module):
    def __init__(self, hidden_size, pad_tag, tag_class):
        super(Decoder, self).__init__()
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.pad_tag = pad_tag

        self.lstm = nn.LSTM(input_size=hidden_size * 6, 
                            hidden_size=hidden_size, 
                            num_layers=self.num_layers, 
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, tag_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch = x.size(0)
        length = x.size(1)
        init_out = torch.zeros(batch, 1, self.hidden_size * 2).to(device)
        hidden_state = (torch.zeros(2 * self.num_layers, 1, self.hidden_size).to(device), 
                        torch.zeros(2 * self.num_layers, 1, self.hidden_size).to(device))

        mem = torch.zeros(batch, self.pad_tag, self.hidden_size * 2).to(device)
        x = torch.cat((x, mem), dim=-1)
        x = x.transpose(1, 0)  
        x = self.dropout(x)
        all_out = []
        for i in range(length):
            if i == 0:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), init_out), dim=-1), hidden_state)
            else:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), out), dim=-1), hidden_state)
            all_out.append(out)
        output = torch.cat(all_out, dim=1) 
        res = self.fc(output)
        return res 

