import argparse
import logging
from functools import partial
from tkinter import W
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DataCollatorWithPadding,
    SchedulerType,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="./data/slot/test.json")
    parser.add_argument("--target_dir", type=str, default="./ckpt/slot/")
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, default="./pred_s.csv")
    args = parser.parse_args()
    
    return args

def main(args):
    accelerator = Accelerator()  
    config = AutoConfig.from_pretrained(args.target_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.target_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.target_dir, config=config)
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    args.token_col, args.tag_col = "tokens", "tags"
    idx2tag = config.id2label
    tag2idx = config.label2id
    test_examples = raw_datasets["test"]

    def tokenize_and_align_labels(examples):
        tokenized_examples = tokenizer(examples[args.token_col], is_split_into_words=True)
    
        if examples.get(args.tag_col):
            tags = examples[args.tag_col]
        labels = []
        for i in range(len(tokenized_examples["input_ids"])):
            word_ids = tokenized_examples.word_ids(batch_index=i)
            prev = None
            example_lbl = []
            for word_id in word_ids:
                if word_id == None:
                    example_lbl.append(-1) 
                elif word_id != prev:   
                    if examples.get(args.tag_col):
                        example_lbl.append(tag2idx[tags[i][word_id]])
                    else:
                        example_lbl.append(0)  
                else:
                    example_lbl.append(-1)
                prev = word_id
            labels.append(example_lbl)
        tokenized_examples["labels"] = labels

        return tokenized_examples



    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            desc="Running tokenizer on dataset",
        )
    test_dataset = processed_raw_datasets["test"]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)   
    model.eval()
    all_predictions = []
    for data in test_dataloader:
        with torch.no_grad():
            outputs = model(**data)
            predictions = accelerator.gather(outputs.logits.argmax(dim=-1))
            references=accelerator.gather(data["labels"])
            predictions = torch.where(references != -1, predictions, references)
            all_predictions += predictions.cpu().tolist()
    pre = [[example_id, ' '.join([idx2tag[tag_id] for tag_id in pred if tag_id != -1])]    \
                            for example_id, pred in zip(test_examples["id"], all_predictions)]
    with open(args.out_file, 'w') as f:
        f.write("id,tags\n")
        for i, l in pre:
            f.write("{},{}\n".format(i, l))

if __name__ == "__main__":
    args = parse_args()
    main(args)