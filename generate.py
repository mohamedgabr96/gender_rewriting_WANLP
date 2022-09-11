from tqdm import tqdm
import numpy as np

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
