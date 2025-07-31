import itertools
import matplotlib.pyplot as plt
import os
import scienceplots
from dotenv import load_dotenv
import gurobipy as gp
from utilities.systemselector import SystemSelectorEnum
from utilities.train_autoencoder import train_autoencoder
from scripts.certify import certify_autoencoder
from scripts.certify_nonlinear import certify_nonlinear
from scripts.certify_hybrid import certify_hybrid
            

if __name__ == '__main__':

    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = {
        "WLSACCESSID": os.getenv("WLSACCESSID"),
        "WLSSECRET": os.getenv("WLSSECRET"),
        "LICENSEID": int(os.getenv("LICENSEID"))
    }

    cases = {"1": "linear", "2": "pwa", "3": "nonlinear"}

    system_map = {
            "1": SystemSelectorEnum.linear,
            "2": SystemSelectorEnum.pwa,
            "3": SystemSelectorEnum.nonlinear
    }


    for key in cases:
 
        results_dir = f"certification_results/{cases[key]}"
        os.makedirs(results_dir, exist_ok=True)

        layers_list = [2]
        neurons_list = [6]
        nx = 3
        mx_factors = [2]
        window = 1

        experiment_summary = []

        with gp.Env(params=params) as env:
            with gp.Model(env=env) as model_gp:
                try:
                    for n_layer, mx, n_neurons in itertools.product(layers_list, mx_factors, neurons_list):
                        state_size = mx * nx
                        print(f"\Model with layers={n_layer}, state size = {state_size}, neurons={n_neurons}")

                        weights_path = f"src/weights/{cases[key]}_weights.weights.h5"

                        sys = system_map[key]

                        model, system, u_n, y_n, u_vn, y_vn = train_autoencoder(fit_horizon = 5,
                                state_reduction = False, state_size = state_size, pairs_iv = 100, affine_struct = False,
                                group_lasso = False, regularizer_weight = 1e-4, output_window_len = 2,
                                n_layer = n_layer, n_neurons = n_neurons, train_ann = False, weights_path=weights_path, sys=sys)
            
                        model_gp.Params.MIPFocus = 1
                        model_gp.Params.Method = 3
                        model_gp.Params.ConcurrentMethod = 3

                        if cases[key] == "linear":

                            solution = certify_autoencoder(
                                        system=system,
                                        model_ann=model,
                                        model_gp=model_gp,
                                        window=window
                            )

                        elif cases[key] == "pwa":

                            solution = certify_hybrid(
                                        system=system,
                                        model_ann=model,
                                        model_gp=model_gp,
                                        window=window
                            )

                        elif cases[key] == "nonlinear":

                            solution = certify_nonlinear(
                                system=system,
                                model_ann=model,
                                model_gp=model_gp,
                                window=window
                            ) 

                        result_entry = {
                            "layers": n_layer,
                            "neurons": n_neurons,
                            "state_size": state_size,
                            "status": "Success" if solution else "Failed",
                            "objective_value": model_gp.ObjVal if model_gp.SolCount > 0 else None,
                            "solving_time": solution["runtime"]
                        }

                        if solution:
                            pparam = dict(xlabel=r"Step $k$")
                            with plt.style.context(["science", "ieee", "no-latex"]):
                                fig, ax = plt.subplots(figsize=(3.3, 1.5))
                                ax.plot(solution["y_sys"], label=r'$y(k)$')

                                last_idx = len(solution["y_pred"]) - 1
                                x_last = last_idx

                                y_sys_last = solution["y_sys"][last_idx]
                                y_pred_last = solution["y_pred"][last_idx]
                                ax.plot(x_last, y_sys_last, 'o', color='black', markersize=3,  label=r'$y_T$')
                                ax.plot(x_last, y_pred_last, 's', color='red', markersize=3, label=r'$\hat{y}_T$')
                                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
                                ax.set_ylim([-16, 16])
                                ax.set(**pparam)
                                plot_path = os.path.join(results_dir, f"plot_L{n_layer}_N{n_neurons}_S{state_size}.pdf")
                                fig.savefig(plot_path, dpi=300)
                                plt.close()
                                result_entry["plot"] = plot_path

                            dat_path = os.path.join(results_dir, f"plot_L{n_layer}_N{n_neurons}_S{state_size}.dat")
                            with open(dat_path, "w") as f_dat:
                                f_dat.write("k\ty_sys\ty_pred\n")
                                for k, (y_s, y_p) in enumerate(zip(solution["y_sys"], solution["y_pred"])):
                                    f_dat.write(f"{k}\t{y_s:.6f}\t{y_p:.6f}\n")
                            result_entry["dat"] = dat_path
                    
                        experiment_summary.append(result_entry)

                except Exception as e:
                    print(f"Experiment failed: {e}")

                txt_summary_path = os.path.join(results_dir, "experiment_summary.txt")

                with open(txt_summary_path, "w") as f:
                    for entry in experiment_summary:
                        f.write(f"Layers: {entry['layers']}, Neurons: {entry['neurons']}, State Size: {entry['state_size']}\n")
                        f.write(f"Status: {entry['status']}\n")
                        f.write(f"Objective Value: {entry['objective_value']}\n")
                        f.write(f"Solving time: {entry['solving_time']}\n")
                        if "plot" in entry:
                            f.write(f"Plot saved at: {entry['plot']}\n")
                        f.write("-" * 60 + "\n")

                print(f"Summary saved to {txt_summary_path}")
        

                    
                

        