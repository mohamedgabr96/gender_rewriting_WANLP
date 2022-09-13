import torch
import os
import yaml
from tqdm import tqdm

def yaml_loader(path):
    with open(path, encoding="utf-8") as yaml_file:
        yaml_loaded = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_loaded


def load_normal_data_speaker_listener(path, split="train", direction="MF", augmented=False):

    # MM: male speaking to a male
    # MF: male speaking to a female 
    # FF: female speaking to a female
    # FM: female speaking to a male

    augmented_string = ".augmented" if augmented else ""

    with open(os.path.join(path, split, f"{split}.arin.tokens" + augmented_string), encoding="utf-8") as file_opened:
        source = file_opened.readlines()

    sentences_list_source = []
    sentence_temp_list = []
    for line in source:
        if line == '\n':
            sentences_list_source.append(sentence_temp_list)
            sentence_temp_list = []
        else:
            temp_line = line.strip("\n").split(" ")
            if len(temp_line[2]) == 2 and (temp_line[2][1] == "M" or temp_line[2][1] == "F"):
                word = temp_line[1]
                label = 1 if temp_line[2][1] == "M" else 2
            else:
                word = temp_line[1]
                label = 0
            sentence_temp_list.append([word, label])
    
    # source = [x.replace("</s>", "").replace("</s>", "").replace("</s>", "") for x in source]

    with open(os.path.join(path, split, f"{split}.ar.{direction}.tokens" + augmented_string), encoding="utf-8") as file_opened:
        target = file_opened.readlines()

    sentences_list_target = []
    sentence_temp_list = []
    for line in target:
        if line == '\n':
            sentences_list_target.append(sentence_temp_list)
            sentence_temp_list = []
        else:
            temp_line = line.strip("\n").split(" ")
            if len(temp_line[2]) == 2 and (temp_line[2][1] == "M" or temp_line[2][1] == "F"):
                word = temp_line[1]
                label = 1 if temp_line[2][1] == "M" else 2
            else:
                word = temp_line[1]
                label = 0
            sentence_temp_list.append([word, label])
    
    return sentences_list_source, sentences_list_target


def tokenize_data(config, data_in, tokenizer, DEBUG=False):
    new_sentence_list = []
    count = 0
    for sentence in tqdm(data_in):
        count += 1
        temp_list_sentence = []
        for word in sentence:
            word_str = word[0]
            tokenized_word = tokenizer([word_str])["input_ids"][0][:-1]
            temp_list_sentence.append([tokenized_word[0], word[1]])
            for subword in tokenized_word[1:]:
                temp_list_sentence.append([subword, 3])
        
        max_len_truncate = config.max_length if tokenizer.eos_token_id is None else  config.max_length - 1
        if len(temp_list_sentence) > max_len_truncate:
            temp_list_sentence = temp_list_sentence[:max_len_truncate]
            if tokenizer.eos_token_id is not None:
                temp_list_sentence.append(tokenizer.eos_token_id)
        else:
            if tokenizer.eos_token_id is not None:
                temp_list_sentence.append([tokenizer.eos_token_id, 3])
            while len(temp_list_sentence) < config.max_length:
                temp_list_sentence.append([tokenizer.pad_token_id, 3])
        
        assert len(temp_list_sentence) == config.max_length
        new_sentence_list.append(temp_list_sentence)

        if DEBUG and count > 100:
            return new_sentence_list

    return new_sentence_list


def create_dataloader(config, tokenizer, split="train"):
    source_data, target_data_MM = load_normal_data_speaker_listener(config.data_path, split=split, direction="MM")
    _, target_data_MF = load_normal_data_speaker_listener(config.data_path, split=split, direction="MF")
    _, target_data_FM = load_normal_data_speaker_listener(config.data_path, split=split, direction="FM")
    _, target_data_FF = load_normal_data_speaker_listener(config.data_path, split=split, direction="FF")

    all_data_combined = source_data + target_data_MM + target_data_MF + target_data_FM + target_data_FF

    tokenized_data = tokenize_data(config, all_data_combined, tokenizer, DEBUG=False)
    dataset = SpeakerListenerDataset(config, tokenized_data)

    params = {'batch_size': config.batch_size,
          'shuffle': True}

    dataloader = torch.utils.data.DataLoader(dataset, **params)

    return dataloader


class SpeakerListenerDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, config, tokenized_data):
        'Initialization'
        self.tokens_ids = torch.tensor([[x[0] for x in y] for y in tokenized_data]).to(config.device)
        self.labels_ids = torch.tensor([[x[1] for x in y] for y in tokenized_data]).to(config.device)

        assert self.tokens_ids.shape[0] == self.labels_ids.shape[0]
        assert self.tokens_ids.shape[1] == self.labels_ids.shape[1]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.tokens_ids)

  def __getitem__(self, index):
        'Generates one sample of data'

        return self.tokens_ids[index], self.labels_ids[index]


def create_dataloaders(config, tokenizer):
    train_dataloader = create_dataloader(config, tokenizer)
    dev_dataloader = create_dataloader(config, tokenizer, split="dev")
    test_dataloader = create_dataloader(config, tokenizer, split="test")

    return train_dataloader, dev_dataloader, test_dataloader


