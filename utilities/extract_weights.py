import numpy as np

def extract_weights_encoder(encoder_network):
    encoder_weights = {}
    encoder_weights_dic = {l.name:l for l in encoder_network.layers if l.name.startswith("enc")}
    encoder_weights_dic_names = list(encoder_weights_dic.keys())
    encoder_weights_dic_names.sort()
    for i, layer_name in enumerate(encoder_weights_dic_names):
        layer = encoder_weights_dic[layer_name]
        layer_weights = layer.get_weights()
        if layer_weights:
            encoder_weights[f'layer_{i}'] = {
                'W': layer_weights[0],  # Kernel weights for state/input connections
                'b': layer_weights[1]  # Bias term if it exists
            }

    return encoder_weights

def extract_weights_bridge(bridge_network):
    bridge_weights = {}
    bridge_weights_dic = {l.name:l for l in bridge_network.layers if l.name.startswith("bridge")}
    bridge_weights_dic_names = list(bridge_weights_dic.keys())
    bridge_weights_dic_names.sort()
    for i, layer_name in enumerate(bridge_weights_dic_names):
        layer = bridge_weights_dic[layer_name]
        layer_weights = layer.get_weights()
        if layer_weights:
            bridge_weights[f'layer_{i}'] = {
                'W': layer_weights[0],  # Kernel weights for state/input connections
                'b': layer_weights[1]  # Bias term if it exists
            }

    return bridge_weights

def extract_weights_decoder(decoder_network):
    decoder_weights = {}
    decoder_weights_dic = {l.name:l for l in decoder_network.layers if l.name.startswith("dec")}
    decoder_weights_dic_names = list(decoder_weights_dic.keys())
    decoder_weights_dic_names.sort()
    for i, layer_name in enumerate(decoder_weights_dic_names):
        layer = decoder_weights_dic[layer_name]
        layer_weights = layer.get_weights()
        if layer_weights:
            decoder_weights[f'layer_{i}'] = {
                'W': layer_weights[0],  # Kernel weights for state connections
                'b': layer_weights[1]  # Bias term if it exists
            }

    return decoder_weights