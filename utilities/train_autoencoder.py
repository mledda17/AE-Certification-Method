from src.model.autoencoder import Autoencoder

def train_autoencoder(fit_horizon: int = 5, 
                      state_reduction: bool = False,
                      state_size: int = 6,
                      pairs_iv: int = 10,
                      affine_struct: bool = False,
                      group_lasso: bool = False,
                      regularizer_weight: float = 1e-4,
                      output_window_len: int = 2,
                      n_layer: int = 2,
                      n_neurons: int = 6,
                      train_ann: bool = False, 
                      weights_path = None,
                      sys = None):


    system, u_n, y_n, u_vn, y_vn = sys()

    model = Autoencoder(affine_struct=affine_struct, group_lasso=group_lasso,
                        state_reduction=state_reduction, fit_horizon=fit_horizon,
                        stride_len=pairs_iv, output_window_len=output_window_len, n_layer=n_layer,
                        n_neurons=n_neurons, regularizer_weight=regularizer_weight, state_size=state_size)
    
    model.set_dataset(u_n.copy(), y_n.copy(), u_vn.copy(), y_vn.copy())

    if train_ann:
        model.fit_model()
        model.model.save_weights(weights_path)
        print(f"Weights saved in {weights_path}")
    else:
        model.model, model.encoder_network, model.decoder_network, model.bridge_network = model.ann_model()
        model.model.load_weights(weights_path)

    return model, system, u_n, y_n, u_vn, y_vn