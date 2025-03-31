from utilities.extract_weights import extract_weights_encoder, extract_weights_decoder

def collect_data(system, model):
    encoder_network = model.encoder_network
    decoder_network = model.decoder_network

    nx_ann = model.state_size
    na_ann = 100
    n_neurons = model.n_neurons
    n_layers = model.n_layer

    ny_sys = system.output_size

    # Construct
    encoder_weights = extract_weights_encoder(encoder_network)
    decoder_weights = extract_weights_decoder(decoder_network)

    return encoder_weights, decoder_weights, nx_ann, na_ann, n_neurons, n_layers, ny_sys