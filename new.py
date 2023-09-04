import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import torch

import merge_awq

def load_model_with_cache(model, cache_path):
    if os.path.exists(cache_path):
        model = torch.load(cache_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model)
        torch.save(model, cache_path)
    return model.to(0)


def load_cached_wikitext():
    print('loading dataset')
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    print('tokenizing')
    encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

    return encodings


@torch.no_grad()
def get_ppl(model, tokenizer, num_samples=1000):
    # copied from transformers website
    model.eval()

    encodings = load_cached_wikitext()

    print('about to calculate ppl')
    max_length = 2048
    #stride = 512
    stride = 1024
    device = 0

    count = 0

    if num_samples == -1:
        length = encodings.input_ids.size(1) / stride
    else:
        length = num_samples

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride), total=length):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)
        count += 1

        if num_samples > 0 and count >= num_samples:
            break

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


def apply_awq_quantisation(model):
    tensors = '/code/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/pytorch_model.bin'
    return merge_awq(model, tensors)


if __name__ == '__main__':
    #awq_model = 'abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq'
    #base_model = 'facebook/opt-6.7b'

    ## for OPT
    ## fp16 expect: 12.29
    ## AWQ: 12.44

    ## according to discussions:
    ## for llama-2-7b
    ## expecting like 5.9565
    ## for quantised its more like 6.0863
    ## others say its 5.68

    ## for unquantised
    ## I get tensor(4.8595, device='cuda:0') with stride 512
    ## and with 1024 I get

    awq_model = 'abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq'
    base_model = 'meta-llama/Llama-2-7b-hf'
    model = base_model

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = load_model_with_cache(model, cache_path='model.cache')
    model = model.half()

    #model = apply_awq_quantisation(model)

    tokenizer.pad_token = tokenizer.eos_token 
    p = get_ppl(model, tokenizer, num_samples=-1)
    print('ppl', p)
