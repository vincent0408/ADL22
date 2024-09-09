import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
import torch.nn as nn

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_dl = DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True,
                          pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    eval_dl = DataLoader(dataset=datasets[DEV], batch_size=args.batch_size, shuffle=True,
                         pin_memory=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(embeddings=embeddings,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          num_class=datasets[TRAIN].num_classes).to(args.device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
   
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        correct = 0
        train_total = 0

        for inputs, labels in train_dl:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        print('Epoch {}'.format(epoch))
        print("Train accuracy: {:.3f}".format(correct / train_total))
        print("Train total Loss: {:.3f}".format(epoch_loss))

        model.eval()
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for inputs, labels in eval_dl:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)

                eval_correct += (predicted == labels).sum().item()
                eval_total += labels.size(0)

            eval_acc = eval_correct / eval_total
            print("Eval accuracy: {:.3f}\n".format(eval_acc))

            if(eval_acc > best_acc):
                best_acc = eval_acc
                torch.save(model.state_dict(), args.ckpt_dir /'intent-model.pt')
                print("Saved model at epoch {}\n".format(epoch))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args("")
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
