import os
import sys
import json
import numpy as np
from tqdm import tqdm
from transformers import BartTokenizer
import torch
import torch.nn as nn
import random
import ujson


class MedMentionsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, test_set=False):
        self.encodings = encodings
        self.labels = labels
        self.test_set = test_set

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label_ids"] = torch.tensor(self.labels["labels"][idx])
        item["decoder_input_ids"] = torch.tensor(self.labels["decoder_input_ids"][idx])
        item["decoder_attention_mask"] = torch.tensor(
            self.labels["attention_mask"][idx]
        )
        # the decoder atten mask has the same length as label of decoder input
        if self.test_set:
            item["decoder_input_ids_test"] = torch.tensor(
                self.labels["decoder_input_ids_test"][idx]
            )
            item["decoder_attention_mask_test"] = torch.tensor(
                self.labels["attention_mask_test"][idx]
            )

        # if self.labels['unlikelihood_tokens']:
        #     item['unlikelihood_mask'] = torch.zeros()
        return item

    def __len__(self):
        return len(self.labels["labels"])


def read_file_bylines(file):
    with open(file, "r") as f:
        lines = f.readlines()
    output = []
    for item in lines[:]:
        output.append(item.strip("\n"))
    return output


def encode_data_to_json(dataset_path, tokenizer):
    fi = [dataset_path + f for f in [".source", ".target"]]

    with open(fi[1] + ".token.json", "w") as f:
        for x in tqdm(read_file_bylines(fi[1])):
            xi = ujson.loads(x)
            lines = [
                list(tokenizer(" " + xi[0])["input_ids"])[1:-1],
                list(tokenizer(" " + xi[1])["input_ids"])[1:-1],
            ]
            lines = ujson.dumps(lines, ensure_ascii=False)
            f.write(lines + "\n")

    with open(fi[0] + ".token.json", "w") as f:
        for x in tqdm(read_file_bylines(fi[0])):
            # print(x)

            ix = ujson.loads(x)[0]
            line = list(tokenizer(" " + ix)["input_ids"])[1:-1]
            line = ujson.dumps([line], ensure_ascii=False)
            f.write(line + "\n")


def read_ids_from_json(path, prefix_mention_is=False):
    files = [path + f for f in [".source", ".target"]]
    tokens_x = {"input_ids": [], "attention_mask": []}
    tokens_y = {
        "labels": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "decoder_input_ids_test": [],
        "attention_mask_test": [],
        "unlikelihood_tokens": [],
    }
    max_len_x = 0
    max_len_y = 0
    for x, y in tqdm(
        zip(
            read_file_bylines(files[0] + ".token.json"),
            read_file_bylines(files[1] + ".token.json"),
        )
    ):
        x = ujson.loads(x)
        y = ujson.loads(y)

        prefix = y[0]
        label = y[1]
        y = sum(y, [])

        max_len_x = np.max([max_len_x, len(sum(x, [])) + 3])
        max_len_y = np.max([max_len_y, len(y) + 2])

        tokens_x["input_ids"].append([0] + x[0] + [2])
        tokens_x["attention_mask"].append(list(np.ones_like(tokens_x["input_ids"][-1])))

        if prefix_mention_is:
            tokens_y["decoder_input_ids"].append([2] + y)  # decoder input
            labs_prefix = [-100] * len(prefix) + label + [2]
            tokens_y["labels"].append(labs_prefix)
            assert len(labs_prefix) == len(y) + 1
        else:
            tokens_y["decoder_input_ids"].append([2] + label)  # decoder input
            tokens_y["labels"].append(label + [2])  # labels
        tokens_y["attention_mask"].append(
            list(np.ones_like(tokens_y["decoder_input_ids"][-1]))
        )

        if "test" in path:
            if prefix_mention_is:
                tokens_y["decoder_input_ids_test"].append(prefix)
            else:
                tokens_y["decoder_input_ids_test"].append([2])
            tokens_y["attention_mask_test"].append(
                list(np.ones_like(tokens_y["decoder_input_ids_test"][-1]))
            )

    tokens_x = padding_sequence(tokens_x, max_len_x)
    tokens_y = padding_sequence(tokens_y, max_len_y)

    return tokens_x, tokens_y


def padding_sequence(tokens, max_len):
    for k in tqdm(tokens.keys()):
        for idx in range(len(tokens[k])):
            if k == "input_ids" or k == "decoder_input_ids":
                tokens[k][idx] = list(
                    np.pad(
                        tokens[k][idx],
                        ((0, max_len - len(tokens[k][idx]))),
                        "constant",
                        constant_values=(1, 1),
                    )
                )
            elif k == "attention_mask":
                tokens[k][idx] = list(
                    np.pad(
                        tokens[k][idx],
                        ((0, max_len - len(tokens[k][idx]))),
                        "constant",
                        constant_values=(0, 0),
                    )
                )
            elif k == "labels":
                tokens[k][idx] = np.pad(
                    tokens[k][idx],
                    ((0, max_len - len(tokens[k][idx]))),
                    "constant",
                    constant_values=(1, 1),
                )
                tokens[k][idx][tokens[k][idx] == 1] = -100
                tokens[k][idx] = list(tokens[k][idx].astype(int))
    return tokens


def prepare_trainer_dataset(
    tokenizer, text_path=None, prefix_mention_is=False, evaluate=False
):

    if not evaluate:
        if not os.path.exists(os.path.join(text_path, "train.source.token.json")):
            encode_data_to_json(os.path.join(text_path, "train"), tokenizer)
        train_tokens_x, train_tokens_y = read_ids_from_json(
            os.path.join(text_path, "train"), prefix_mention_is=prefix_mention_is
        )
        train_set = MedMentionsDataset(train_tokens_x, train_tokens_y)
        return train_set, None, None

    else:
        if not os.path.exists(os.path.join(text_path, "dev.source.token.json")):
            encode_data_to_json(os.path.join(text_path, "dev"), tokenizer)
        if not os.path.exists(os.path.join(text_path, "test.source.token.json")):
            encode_data_to_json(os.path.join(text_path, "test"), tokenizer)

        dev_tokens_x, dev_tokens_y = read_ids_from_json(
            os.path.join(text_path, "dev"), prefix_mention_is=prefix_mention_is
        )
        test_tokens_x, test_tokens_y = read_ids_from_json(
            os.path.join(text_path, "test"), prefix_mention_is=prefix_mention_is
        )

        dev_set = MedMentionsDataset(dev_tokens_x, dev_tokens_y, test_set=True)
        test_set = MedMentionsDataset(test_tokens_x, test_tokens_y, test_set=True)

        return None, dev_set, test_set
