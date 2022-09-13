import numpy as np
import os

def load_normal_data_speaker_listener(path, split="train", augmented=False):

    # MM: male speaking to a male
    # MF: male speaking to a female 
    # FF: female speaking to a female
    # FM: female speaking to a male

    augmented_string = ".augmented" if augmented else ""

    with open(os.path.join(path, split, f"{split}.ar.MF.tokens" + augmented_string), encoding="utf-8") as file_opened:
        source = file_opened.readlines()

    sentences_list_source = []
    sentence_temp_list = []
    for line in source:
        if line == '\n':
            sentences_list_source.append(sentence_temp_list)
            sentence_temp_list = []
        else:
            temp_line = line.strip("\n").split(" ")
            if temp_line[2][0] == "1" or temp_line[2][0] == "2":
                word = temp_line[1]
                label = int(temp_line[2][0])
                sentence_temp_list.append([word, label])
    

    with open(os.path.join(path, split, f"{split}.ar.FM.tokens" + augmented_string), encoding="utf-8") as file_opened:
        target = file_opened.readlines()

    sentences_list_target = []
    sentence_temp_list = []
    for line in target:
        if line == '\n':
            sentences_list_target.append(sentence_temp_list)
            sentence_temp_list = []
        else:
            temp_line = line.strip("\n").split(" ")
            if temp_line[2][0] == "1" or temp_line[2][0] == "2":
                word = temp_line[1]
                label = int(temp_line[2][0])
                sentence_temp_list.append([word, label])
    
    return sentences_list_source, sentences_list_target


def create_dict_mapper(data_path, add_dev_test=False):
    # data_path = "/mnt/default/zcodetasks/code/mogabr-codes/proj_data/Arabic-parallel-gender-corpus-v-2.1/data"
    add_dev_test = True

    words_alignments_source, words_alignments_target = load_normal_data_speaker_listener(data_path, split="train", augmented=False)
    words_alignments_source = [item for sublist in words_alignments_source for item in sublist]
    words_alignments_target = [item for sublist in words_alignments_target for item in sublist]

    if add_dev_test:
        words_alignments_source_dev, words_alignments_target_dev = load_normal_data_speaker_listener(data_path, split="dev", augmented=False)
        words_alignments_source_test, words_alignments_target_test = load_normal_data_speaker_listener(data_path, split="test", augmented=False)
        words_alignments_source_dev = [item for sublist in words_alignments_source_dev for item in sublist]
        words_alignments_target_dev = [item for sublist in words_alignments_target_dev for item in sublist]
        words_alignments_source_test = [item for sublist in words_alignments_source_test for item in sublist]
        words_alignments_target_test = [item for sublist in words_alignments_target_test for item in sublist]

        words_alignments_source = words_alignments_source + words_alignments_source_dev + words_alignments_source_test
        words_alignments_target = words_alignments_target + words_alignments_target_dev + words_alignments_target_test


    mapping_dict = {}

    for source_word, target_word in zip(words_alignments_source, words_alignments_target):
        mapping_dict[source_word[0]] = target_word[0]
        mapping_dict[target_word[0]] = source_word[0]

    return mapping_dict
