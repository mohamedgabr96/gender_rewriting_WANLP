import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from argparse import Namespace
import argparse
import os
import sys
sys.path.append("../")
from utils_male_female import create_dataloaders, yaml_loader
import wandb
from tqdm import tqdm, trange

def main(config):
    ### Dataloading

    ## Model Loading
    wandb.init(project="my-gender-rewrite-project-male_female",
            name=f"experiment_{config.exp_name}", 
            # Track hyperparameters and run metadata
            config={
            "learning_rate": config.learning_rate,
            "model_name": config.model_name,
            "speak_listen_mode": config.speak_listen_mode,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "eval_every": config.eval_every,
            "epochs": config.no_epochs,
            })



    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    train_dataloader, dev_dataloader, test_dataloader = create_dataloaders(config, tokenizer)

    label_names = ["NONE", "MALE", "FEMALE", "PAD"]
    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(config.model_name, id2label=id2label, label2id=label2id).to(config.device)

    ## Criterion and Schedulers
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    no_steps = 0
    model.train()
    ## training Loop
    previous_dev_loss = 10000
    for epoch in range(config.no_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                training_loss_val = training_loop(model, batch, optimizer, scheduler)
                no_steps+=1

                wandb.log({"Training Loss": training_loss_val})

                if no_steps % config.eval_every == 0:
                    dev_loss_val, accuracy, accuracy_sl_only = eval_loop(model, dev_dataloader, tokenizer)
                    wandb.log({"Dev Loss": dev_loss_val})
                   
                    wandb.log({"Accuracy": accuracy, "Accuracy Speaker Listener": accuracy_sl_only})

                    if dev_loss_val < previous_dev_loss:
                        save_model(model, config.model_saves_path + "_" + config.exp_name, no_steps)
                        
                    previous_dev_loss = dev_loss_val


                tepoch.set_postfix(loss=training_loss_val)
        


    print("Done")

def training_loop(model, batch, optimizer, scheduler):

    optimizer.zero_grad()

    model_output = model(input_ids=batch[0], labels=batch[1])

    loss = model_output.loss

    loss.backward()
    optimizer.step()

    return loss.item()

def eval_loop(model, dev_loader, tokenizer, generate_sents=False):
    model.eval()
    
    no_batches = 0
    accum_loss = 0

    total_correct_labels = 0
    total_labels =  0

    total_correct_labels_sl_only = 0
    total_labels_sl_only =  0

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

        # Without the None
        labels_to_attend_on = (labels != 3) & (labels != 0)
        correct_labels = labels_to_attend_on & (logits_argmax == labels)
        no_correct_labels_sl_only = int(sum(sum(correct_labels.to(int))).cpu().detach().numpy())
        no_labels_sl_only = int(sum(sum(labels_to_attend_on.to(int))).cpu().detach().numpy())
        
        total_correct_labels_sl_only += no_correct_labels_sl_only
        total_labels_sl_only +=  no_labels_sl_only

    accuracy_all = total_correct_labels / total_labels
    accuracy_sl_only = total_correct_labels_sl_only / total_labels_sl_only

    model.train()

    return accum_loss / no_batches, accuracy_all, accuracy_sl_only


def save_model(model, path, step_no):
    path_with_step = os.path.join(path, "dev_" + str(step_no))
    if not os.path.exists(path_with_step):
        os.makedirs(path_with_step)

    model.save_pretrained(path_with_step)




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