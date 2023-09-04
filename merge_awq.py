from functools import partial
import torch
import time

import os
import torch
import qmodule

# llama 7B
NUM_LAYERS = 32


def quantise_layers(raw_model, weights_path):
    print('loading the GPTQ weights')
    print('loading', weights_path)
    gptq_tensors = torch.load(weights_path)

    gb_in_bytes = 1024 ** 3
    before = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY BEFORE (GB)', before)

    quantise_single_layer(raw_model, gptq_tensors, 'mlp.down_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'mlp.up_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'mlp.gate_proj')

    load_weights(raw_model, gptq_tensors, 'input_layernorm')
    load_weights(raw_model, gptq_tensors, 'post_attention_layernorm')

    quantise_single_layer(raw_model, gptq_tensors, 'self_attn.o_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'self_attn.k_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'self_attn.v_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'self_attn.q_proj')

    after = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY AFTER (GB)', after)
    print('% decreases in memory', '{:.4f}'.format(100 * (before - after) / before))

    print('####', raw_model, '####')
    print(raw_model)

    return raw_model


def get_linear(qweight, qzeros, scales, in_features, out_features):
    layer = qmodule.WQLinear(
        w_bit=4,
        group_size=128,
        in_features=in_features,
        out_features=out_features,
        bias=None,
        dev=0
    )

    assert layer.qweight.shape == qweight.shape
    assert layer.qweight.dtype == qweight.dtype
    layer.qweight = qweight

    assert layer.qzeros.shape == qzeros.shape
    assert layer.qzeros.dtype == qzeros.dtype
    layer.qzeros = qzeros

    assert layer.scales.shape == scales.shape
    assert layer.scales.dtype == scales.dtype
    layer.scales = scales

    #import pdb; pdb.set_trace()

    return layer


def quantise_multiple_layers(raw_model, gptq_tensors, names, output_name):
    print('quantising {} to {}...'.format(','.join(names), output_name))
    for pos in range(0, NUM_LAYERS):
        name = 'model.layers.{}.{}'.format(pos, output_name)
        quant_layer = get_multiple_quant_layer(gptq_tensors, pos, names)
        parent = '.'.join(name.split('.')[1:-1])
        key = output_name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def quantise_single_layer(raw_model, gptq_tensors, name):
    print('quantising {}....'.format(name))
    for pos in range(0, NUM_LAYERS):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        quant_layer = get_quant_layer(gptq_tensors, pos, name=name)
        parent = '.'.join(target_name.split('.')[1:-1])
        key = name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def load_weights(raw_model, gptq_tensors, name):
    print('load weights {}....'.format(name))
    for pos in range(0, NUM_LAYERS):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        weights = gptq_tensors[target_name + '.weight']
        path = '.'.join(target_name.split('.')[1:])
        setattr(raw_model.get_submodule(path), 'weight', torch.nn.Parameter(weights))
    return raw_model


def calculate_memory_usage(model):
    # doesnt include the peak usage for the forward pass
    # returns in bytes
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs
