import os
import yaml
import torch

def yaml_loader(path):
    with open(path, encoding="utf-8") as yaml_file:
        yaml_loaded = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_loaded


def load_normal_data(path, split="train", direction="MF", augmented=False):

    # MM: male speaking to a male
    # MF: male speaking to a female 
    # FF: female speaking to a female
    # FM: female speaking to a male

    augmented_string = ".augmented" if augmented else ""

    with open(os.path.join(path, split, f"{split}.arin" + augmented_string), encoding="utf-8") as file_opened:
        source = file_opened.readlines()
    
    source = [x.replace("</s>", "").replace("</s>", "").replace("</s>", "") for x in source]

    with open(os.path.join(path, split, f"{split}.ar.{direction}" + augmented_string), encoding="utf-8") as file_opened:
        target = file_opened.readlines()

    target = [x.replace("</s>", "").replace("</s>", "").replace("</s>", "") for x in target]

    assert len(source) == len(target)
    
    return source, target


def tokenize_data(config, data_in, tokenizer):
    data_out = tokenizer(data_in, return_tensors="pt", padding="max_length", max_length=config.max_length)
    return data_out

def create_dataloader(config, tokenizer, split="train"):
    source_data, target_data = load_normal_data(config.data_path, split=split, direction=config.speak_listen_mode)
    tokenized_source = tokenize_data(config, source_data, tokenizer)
    tokenized_target = tokenize_data(config, target_data, tokenizer)
    dataset = GenderDataset(config, tokenized_source, tokenized_target, tokenizer.pad_token_id)

    shuffle_param = True if split == "train" else False
    params = {'batch_size': config.batch_size if split == "train" else config.batch_size*2,
          'shuffle': shuffle_param, 'num_workers': config.num_workers}

    dataloader = torch.utils.data.DataLoader(dataset, **params)

    return dataloader


def create_dataloaders(config, tokenizer):
    train_dataloader = create_dataloader(config, tokenizer)
    dev_dataloader = create_dataloader(config, tokenizer, split="dev")
    test_dataloader = create_dataloader(config, tokenizer, split="test")

    return train_dataloader, dev_dataloader, test_dataloader


class GenderDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, config, source_data, target_data, pad_token_id):
        'Initialization'
        self.source_ids = source_data.input_ids
        self.target_ids = target_data.input_ids
        self.target_ids = torch.tensor([
                [(l if l != pad_token_id else -100) for l in label] for label in self.target_ids
            ])
        self.source_masks = source_data.attention_mask
        self.target_masks = target_data.attention_mask

        assert self.source_ids.shape[0] == self.target_ids.shape[0] == self.source_masks.shape[0] == self.target_masks.shape[0]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source_ids)

  def __getitem__(self, index):
        'Generates one sample of data'

        return self.source_ids[index], self.target_ids[index], self.source_masks[index], self.target_masks[index]

def uniq(seq, idfun=None):
    # order preserving                                                                          
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:                                                               
        # if seen.has_key(marker)                                                               
        # but in new ones:                                                                      
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.                                
        # Let's try using the fastest.                                                          
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode)


def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)

    