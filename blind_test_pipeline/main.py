import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
import sys
import torch
import wandb
from tqdm import tqdm
import json
import numpy as np
from argparse import Namespace
import argparse
import yaml
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from alignment_dict.dict_aligner import create_dict_mapper


def yaml_loader(path):
    with open(path, encoding="utf-8") as yaml_file:
        yaml_loaded = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_loaded

def main(config):
    model_name_tokenizer = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    path_gender_identifier= "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/save_folder_male_female_male_female-camelbertmsa/dev_167000"
    path_speaker_listener_identifier= "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/save_folder_speaker_listener_speaker_listener_camelbertmsa/dev_133000"
    tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer, use_fast=False)
    gender_model = AutoModelForTokenClassification.from_pretrained(path_gender_identifier).to("cuda")
    speaker_listener_model = AutoModelForTokenClassification.from_pretrained(path_speaker_listener_identifier).to("cuda")

    id2label_gender = {
            0: "NONE",
            1: "MALE",
            2: "FEMALE",
            3: "PAD"
        }

    id2label_speaker_listener = {
            0: "NONE",
            1: "SPEAKER",
            2: "LISTENER",
            3: "PAD"
        }

    data_path = "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/test.txt"
    source_data = load_normal_data_speaker_listener(data_path)

    tokenized_data = tokenize_data(config, source_data, tokenizer, DEBUG=False)
    dataset = SpeakerListenerDataset(config, tokenized_data)
    params = {'batch_size': config.batch_size,
          'shuffle': False}

    path_to_generated_data = "/mnt/default/zcodetasks/code/mogabr-codes/proj_git/gender_rewriting_WANLP/blind_test_pipeline/results_final_generation/output22_MM.txt"
    with open(path_to_generated_data, encoding="utf-8") as file_opened:
        readdd = file_opened.readlines()
        readdd = [x.strip("\n") for x in readdd]
    seq2seq_data_generation = readdd

    dev_dataloader = torch.utils.data.DataLoader(dataset, **params)

    _, _, _, gender_logits, labels_attend_on_gender = eval_loop_tokens(gender_model, dev_dataloader, tokenizer)
    _, _, _, speak_listen_logits, labels_attend_on_speak_listen = eval_loop_tokens(speaker_listener_model, dev_dataloader, tokenizer)
    normal_sentences_dev = [[x[0] for x in y] for y in source_data]
    # Create aligner 
    alignment_dict = create_dict_mapper(config.data_path, aug=False)

    ## Get the stuff that needs to change based on the direction
    counter = 0
    counter_fallback_words = 0
    total_number_of_words = 0
    counter_fallback_seq2seq = 0
    all_new_sentences = []
    wanted_direction = config.speak_listen_mode
    dict_opposite_direction = {"M": "FEMALE", "F": "MALE"}
    for batch in dev_dataloader:
        input_ids = batch[0]
        for sent in input_ids:
            sent_text = normal_sentences_dev[counter]
            only_relevant_labels_gender = gender_logits[counter][labels_attend_on_gender[counter]]
            only_relevant_labels_speak_listen = speak_listen_logits[counter][labels_attend_on_speak_listen[counter]]
            only_relevant_labels_gender = [id2label_gender[x] for x in only_relevant_labels_gender]
            only_relevant_labels_speak_listen = [id2label_speaker_listener[x] for x in only_relevant_labels_speak_listen]
            combined_both = [x + "-" + y for x, y in zip(only_relevant_labels_speak_listen, only_relevant_labels_gender)]

            word_combinations_to_change = [f"SPEAKER-{dict_opposite_direction[wanted_direction[0]]}", f"LISTENER-{dict_opposite_direction[wanted_direction[1]]}"]

            new_sentence = []

            for word_index, word in enumerate(combined_both):
                if word in word_combinations_to_change:
                    text_word = sent_text[word_index]
                    changed_word = alignment_dict.get(text_word, None)
                    changed_word = None if config.remove_dict else changed_word
                    if changed_word is None:
                        new_word = "dummy" ## Fallback not implemented yet
                        res1, res2 = search_with_prefix(text_word, seq2seq_data_generation[counter])
                        new_word = res2[0] if len(res2) > 0 else "dummy"
                        counter_fallback_words += 1 if new_word == "dummy" else 0
                        if new_word != "dummy":
                            counter_fallback_seq2seq += 1
                        else:
                            new_word = text_word
                    else:
                        new_word = changed_word

                else:
                    new_word = sent_text[word_index]

                new_sentence.append(new_word)
                total_number_of_words += 1


            counter += 1
            all_new_sentences.append(" ".join(new_sentence))

    path_to_file = os.path.join(config.output_save_location, f"arin.to.{config.speak_listen_mode}")

    if not os.path.exists(config.output_save_location):
        os.makedirs(config.output_save_location)


    with open(path_to_file, encoding="utf-8", mode="w") as file_opened:
        for sent in all_new_sentences:
            file_opened.write(sent + "\n")

def load_normal_data_speaker_listener(path):
    with open(os.path.join(path,), encoding="utf-8") as file_opened:
        source = file_opened.readlines()

    sentences_list_source = []
    sentence_temp_list = []
    for line in source:
        sentence_temp_list = []
        line = line.strip("\n").replace("</s>", "").replace("<s>", "")
        words = line.split(" ")
        for word in words:
            sentence_temp_list.append([word, 0])
        sentences_list_source.append(sentence_temp_list)


    return sentences_list_source

def search_with_prefix(word, sentence):
    K = 3
    k_len_subs = [word[i: j] for i in range(len(word)) for j in range(i + 1, len(word) + 1) if len(word[i:j]) == K]
    l = sentence.split(" ")
    result1 = []
    result2 = []
    for k_len_sub in k_len_subs:
        result1 += list(filter(lambda x: x.startswith(k_len_sub), l))

        result2 += list(filter(lambda x: k_len_sub in x, l))

    return result1, result2

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


def eval_loop_tokens(model, dev_loader, tokenizer, generate_sents=False):
    model.eval()
    
    no_batches = 0
    accum_loss = 0

    total_correct_labels = 0
    total_labels =  0

    total_correct_labels_sl_only = 0
    total_labels_sl_only =  0
    list_of_logits = None
    list_labels_to_attend_on = None
    for batch in tqdm(dev_loader): 
        model_output = model(input_ids=batch[0], labels=batch[1])
        no_batches += 1
        accum_loss += model_output.loss.item()
        
        logits_argmax = torch.argmax(model_output.logits, axis=2)
        labels=batch[1]

        ## With the None
        labels_to_attend_on = labels != 3
        correct_labels = labels_to_attend_on & (logits_argmax == labels)
        no_correct_labels = int(sum(sum(correct_labels.to(int))).cpu().detach().numpy())
        no_labels = int(sum(sum(labels_to_attend_on.to(int))).cpu().detach().numpy())

        total_correct_labels += no_correct_labels
        total_labels += no_labels

        if list_labels_to_attend_on is None:
            list_labels_to_attend_on = labels_to_attend_on.cpu().detach().numpy()
        else:
            list_labels_to_attend_on = np.concatenate((list_labels_to_attend_on, labels_to_attend_on.cpu().detach().numpy()), axis=0)

        # Without the None
        labels_to_attend_on = (labels != 3) & (labels != 0)
        correct_labels = labels_to_attend_on & (logits_argmax == labels)
        no_correct_labels_sl_only = int(sum(sum(correct_labels.to(int))).cpu().detach().numpy())
        no_labels_sl_only = int(sum(sum(labels_to_attend_on.to(int))).cpu().detach().numpy())
        
        total_correct_labels_sl_only += no_correct_labels_sl_only

        if list_of_logits is None:
            list_of_logits = logits_argmax.cpu().detach().numpy()
        else:
            list_of_logits = np.concatenate((list_of_logits, logits_argmax.cpu().detach().numpy()), axis=0)


    accuracy_all = total_correct_labels / total_labels

    model.train()

    return accum_loss / no_batches, accuracy_all, 0, list_of_logits, list_labels_to_attend_on


def tokenize_data(config, data_in, tokenizer, DEBUG=False):
    new_sentence_list = []
    count = 0
    for sentence in tqdm(data_in):
        count += 1
        temp_list_sentence = []
        temp_list_sentence.append([tokenizer.cls_token_id, 3])
        for word in sentence:
            word_str = word[0]
            tokenized_word = tokenizer([word_str])["input_ids"][0][1:-1]
            temp_list_sentence.append([tokenized_word[0], word[1]])
            for subword in tokenized_word[1:]:
                temp_list_sentence.append([subword, 3])
        
        max_len_truncate = config.max_length if tokenizer.eos_token_id is None else  config.max_length - 1
        if len(temp_list_sentence) > max_len_truncate:
            temp_list_sentence = temp_list_sentence[:max_len_truncate]
            if tokenizer.eos_token_id is not None:
                if config.model_name == 'CAMeL-Lab/bert-base-arabic-camelbert-msa':
                    temp_list_sentence.append(tokenizer.sep_token_id)
                temp_list_sentence.append(tokenizer.eos_token_id)
        else:
            if tokenizer.eos_token_id is not None:
                if config.model_name == 'CAMeL-Lab/bert-base-arabic-camelbert-msa':
                    temp_list_sentence.append([tokenizer.sep_token_id, 3])
                temp_list_sentence.append([tokenizer.eos_token_id, 3])
            while len(temp_list_sentence) < config.max_length:
                temp_list_sentence.append([tokenizer.pad_token_id, 3])
        
        assert len(temp_list_sentence) == config.max_length
        new_sentence_list.append(temp_list_sentence)

        if DEBUG and count > 100:
            return new_sentence_list

    return new_sentence_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
            type=str,
        default=None,
        help='The path to the config file'
    )

    args = parser.parse_args()

    config_path = args.config_path
    config = Namespace(**yaml_loader(config_path))
    main(config)
