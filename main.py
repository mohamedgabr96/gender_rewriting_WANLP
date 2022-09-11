import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import Namespace
import argparse
import os
from generate import generate
from utils import yaml_loader, create_dataloaders
import wandb
from m2_calculation import calculate_m2
from tqdm import tqdm, trange

def main(config):
    ### Dataloading

    ### Model Loading
    wandb.init(project="my-gender-rewrite-project",
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

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(config.device)

    ## Criterion and Schedulers
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion = torch.nn.NLLLoss()

    no_steps = 0
    model.train()
    ## training Loop
    previous_dev_loss = 10000
    for epoch in range(config.no_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                training_loss_val = training_loop(model, batch, optimizer, criterion, scheduler)
                no_steps+=1

                wandb.log({"Training Loss": training_loss_val})

                if no_steps % config.eval_every == 0:
                    dev_loss_val, output_generation = eval_loop(model, dev_dataloader, tokenizer, generate_sents=config.calc_score_at_eval)
                    wandb.log({"Dev Loss": dev_loss_val})
                    if config.calc_score_at_eval:
                        p, r, f1 = calculate_m2(output_generation, config.m2_edits_path, split='dev', specific_direction=config.speak_listen_mode)

                    print("Precision: " + str(p))
                    print("Recall: " + str(r))
                    print("F-0.5: " + str(f1))
                    print("Dev Loss: " + str(dev_loss_val))

                    wandb.log({"Precision": p, "Recall": r, "F-0.5": f1})

                    if dev_loss_val < previous_dev_loss:
                        save_model(model, config.model_saves_path + "_" + config.exp_name, no_steps)
                        save_generated_output(output_generation, config.model_saves_path + "_" + config.exp_name, no_steps)
                        
                    previous_dev_loss = dev_loss_val


                tepoch.set_postfix(loss=training_loss_val)
        


    print("Done")

def training_loop(model, batch, optimizer, criterion, scheduler):

    optimizer.zero_grad()

    model_output = model(input_ids=batch[0], attention_mask=batch[2], labels=batch[1])

    loss = model_output.loss

    loss.backward()
    optimizer.step()

    return loss.item()

def eval_loop(model, dev_loader, tokenizer, generate_sents=False):
    model.eval()
    
    no_batches = 0
    accum_loss = 0
    for batch in tqdm(dev_loader): 
        model_output = model(input_ids=batch[0], attention_mask=batch[2], labels=batch[1])
        no_batches += 1
        accum_loss += model_output.loss.item()

    output_generation = None
    if generate_sents:
        output_generation = generate(model, dev_loader, tokenizer)


    model.train()

    return accum_loss / no_batches, output_generation


def save_model(model, path, step_no):
    path_with_step = os.path.join(path, "dev_" + str(step_no))
    if not os.path.exists(path_with_step):
        os.makedirs(path_with_step)

    model.save_pretrained(path_with_step)

def save_generated_output(generated_data, path, step_no):
    path_with_step = os.path.join(path, "dev_" + str(step_no))
    if not os.path.exists(path_with_step):
        os.makedirs(path_with_step)

    path_with_file = os.path.join(path_with_step, "generated_dev_data.txt")
    with open(path_with_file, encoding="utf-8", mode="w") as file_opened:
        for line in generated_data:
            file_opened.write(line + "\n")


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