from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

def generate(model, dataloader, tokenizer, DEBUG=False):
    generated_data = None
    counter = 0
    for batch in tqdm(dataloader):
        model_output = model.generate(input_ids=batch[0], max_length=32)
        temp_decoded_output = tokenizer.batch_decode(model_output.cpu().detach().numpy(), skip_special_tokens=True)
        if generated_data is None:
            generated_data = temp_decoded_output
        else:
            generated_data = np.concatenate((generated_data, temp_decoded_output), axis=0)

        counter += 1
        if DEBUG and counter > 15:
            break

    return generated_data

def tokenize_data(data_in, tokenizer):
    data_out = tokenizer(data_in, return_tensors="pt", padding="max_length", max_length=128)
    return data_out

class GenderDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, source_data, target_data, pad_token_id):
        'Initialization'
        self.source_ids = source_data.input_ids.to("cuda")
        self.target_ids = target_data.input_ids
        self.target_ids = torch.tensor([
                [(l if l != pad_token_id else -100) for l in label] for label in self.target_ids
            ]).to("cuda")
        self.source_masks = source_data.attention_mask.to("cuda")
        self.target_masks = target_data.attention_mask.to("cuda")

        assert self.source_ids.shape[0] == self.target_ids.shape[0] == self.source_masks.shape[0] == self.target_masks.shape[0]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source_ids)

  def __getitem__(self, index):
        'Generates one sample of data'

        return self.source_ids[index], self.target_ids[index], self.source_masks[index], self.target_masks[index]


direction = "FM"
model_name = "/mnt/default/zcodetasks/code/mogabr-codes/proj_git_2/gender_rewriting_WANLP/save_folder_FM_arat5msa_seq2seq_augmented/dev_57000"
path = "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/test.txt"
with open(path, encoding="utf-8") as file_opened:
        source = file_opened.readlines()
    
source = [x.replace("</s>", "").replace("</s>", "").replace("<s>", "").strip("\n") for x in source]

source_data = target_data = source

tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5-msa-base", use_fast=False)


tokenized_source = tokenize_data(source_data, tokenizer)
tokenized_target = tokenize_data(target_data, tokenizer)
dataset = GenderDataset(tokenized_source, tokenized_target, tokenizer.pad_token_id)

params = {'batch_size': 16,
        'shuffle': False}

dataloader = torch.utils.data.DataLoader(dataset, **params)


model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

output_generation = generate(model, dataloader, tokenizer)

path_to_output = "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/blind_test_pipeline/results_final_generation"

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)


with open(os.path.join(path_to_output, "output222_" + direction + ".txt"), encoding="utf-8", mode="w") as file_opened:
    for sent in output_generation:
        file_opened.write(sent + "\n")