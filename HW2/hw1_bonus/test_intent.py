import argparse
from functools import partial
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DataCollatorWithPadding,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,

)

def prepare_features(examples, args, tokenizer, intent2id):
    tokenized_examples = tokenizer(examples[args.text_col])
    if examples.get(args.intent_col):
        tokenized_examples["labels"] = [intent2id[intent] for intent in examples[args.intent_col]]
    return tokenized_examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", 
                        type=str, 
                        default="./data/intent/test.json")
    parser.add_argument("--target_dir", 
                        type=str, 
                        default="./ckpt/intent/")
    parser.add_argument("--test_batch_size", 
                        type=int, 
                        default=16)
    parser.add_argument("--out_file", 
                        type=str, 
                        default="./pred_i.csv")
    args = parser.parse_args()
    
    return args

def main(args):
    accelerator = Accelerator()
    config = AutoConfig.from_pretrained(args.target_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.target_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.target_dir, config=config)
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    args.text_col, args.intent_col = "text", "intent"
    intent2idx = config.label2id
    idx2intent = config.id2label
    test_examples = raw_datasets["test"]
    padding = "max_length"
    def preprocess_function(examples):
        texts = (
            (examples["text"],)
        )
        result = tokenizer(*texts, padding=padding, max_length=128, truncation=True)

        if "intent" in examples:
            result["labels"] = [intent2idx[label] for label in examples["intent"]]
        return result
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            desc="Running tokenizer on dataset",
        )
    test_dataset = processed_datasets["test"]
    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    model.eval()
    all_predictions = []
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions += accelerator.gather(predictions).cpu().tolist()
    pre = [[id,idx2intent[pred]] for id, pred in zip(test_examples["id"], all_predictions)]
    with open(args.out_file, 'w') as f:
        f.write("id,intent\n")
        for pred in pre:
            f.write("{},{}\n".format(pred[0], pred[1]))

if __name__ == "__main__":
    args = parse_args()
    main(args)