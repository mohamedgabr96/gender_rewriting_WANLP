from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
import sys
import torch
import wandb
import numpy as np
from argparse import Namespace
import argparse
sys.path.append(os.path.dirname(sys.path[0]))
from utils import yaml_loader
from speaker_listener_classifier.utils_speaker_listener import create_dataloaders as create_dataloaders_splis
from speaker_listener_classifier.utils_speaker_listener import load_normal_data_speaker_listener
from male_female_classifier.utils_male_female import create_dataloaders as create_dataloaders_gender
from tqdm import tqdm
from alignment_dict.dict_aligner import create_dict_mapper

def main(config):
    ### Dataloading

    ### Model Loading
    # wandb.init(project="my-gender-rewrite-project-full_pipeline",
    #         name=f"experiment_{config.exp_name}", 
    #         # Track hyperparameters and run metadata
    #         config={
    #         "model_name": config.model_name,
    #         "speak_listen_mode": config.speak_listen_mode,
    #         "max_length": config.max_length,
    #         "batch_size": config.batch_size,
    #         "eval_every": config.eval_every,
    #         })



    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    _, dev_dataloader_gender, _ = create_dataloaders_gender(config, tokenizer, dont_train=True)
    _, dev_dataloader_speak_listen, _ = create_dataloaders_splis(config, tokenizer, dont_train=True)

    normal_sentences_dev, _ = load_normal_data_speaker_listener(config.data_path, split="dev", direction=config.speak_listen_mode)
    normal_sentences_dev = [[x[0] for x in y] for y in normal_sentences_dev]

    seq2seq_data_generation = load_seq2seq_generation(config.path_to_s2s_generation)

    gender_model = AutoModelForTokenClassification.from_pretrained(config.path_gender_identifier).to(config.device)
    speaker_listener_model = AutoModelForTokenClassification.from_pretrained(config.path_speaker_listener_identifier).to(config.device)

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

    _, _, _, gender_logits, labels_attend_on_gender = eval_loop_tokens(gender_model, dev_dataloader_gender, tokenizer)
    _, _, _, speak_listen_logits, labels_attend_on_speak_listen = eval_loop_tokens(speaker_listener_model, dev_dataloader_speak_listen, tokenizer)

    # Create aligner 
    alignment_dict = create_dict_mapper(config.data_path)

    ## Get the stuff that needs to change based on the direction
    counter = 0
    counter_fallback_words = 0
    total_number_of_words = 0
    all_new_sentences = []
    wanted_direction = config.speak_listen_mode
    dict_opposite_direction = {"M": "FEMALE", "F": "MALE"}
    for batch in dev_dataloader_gender:
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
                    if changed_word is None:
                        new_word = "dummy" ## Fallback not implemented yet
                        res1, res2 = search_with_prefix(text_word, seq2seq_data_generation[counter])
                        new_word = res2[0] if len(res2) > 0 else "dummy"
                        counter_fallback_words += 1 if new_word == "dummy" else 0
                    else:
                        new_word = changed_word

                else:
                    new_word = sent_text[word_index]

                new_sentence.append(new_word)
                total_number_of_words += 1


            counter += 1
            all_new_sentences.append(new_sentence)


    print("Done")


    

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
        total_labels_sl_only +=  no_labels_sl_only

        if list_of_logits is None:
            list_of_logits = logits_argmax.cpu().detach().numpy()
        else:
            list_of_logits = np.concatenate((list_of_logits, logits_argmax.cpu().detach().numpy()), axis=0)


    accuracy_all = total_correct_labels / total_labels
    accuracy_sl_only = total_correct_labels_sl_only / total_labels_sl_only

    model.train()

    return accum_loss / no_batches, accuracy_all, accuracy_sl_only, list_of_logits, list_labels_to_attend_on


def load_seq2seq_generation(path_to_file):
    with open(path_to_file, encoding="utf-8") as file_opened:
        lines_read = file_opened.readlines()
    
    lines_read = [x.strip("\n") for x in lines_read]

    return lines_read

def search_with_prefix(word, sentence):
    l = sentence.split(" ")
    result1 = list(filter(lambda x: x.startswith(word), l))

    result2 = list(filter(lambda x: word in x, l))

    return result1, result2

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
