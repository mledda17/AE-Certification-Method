import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB 
from utilities.collect_data import collect_data


def certify_nonlinear(system, model_ann, model_gp, window):
    encoder_weights, decoder_weights, nx_ann, na_ann, n_neurons, n_layers, ny_sys = collect_data(system, model_ann)

    # Autoencoder variables
    y_k, h_d, x_k, h_e, s, delta = {}, {}, {}, {}, {}, {}  

    x1 = model_gp.addMVar(na_ann+1, name=f"x1")
    x2 = model_gp.addMVar(na_ann+1, name=f"x2")
    inflow = model_gp.addMVar(na_ann+1, name=f"inflow")
    sqrt_x1 = model_gp.addMVar(na_ann+1, lb=0, name="sqrt_x1")
    sqrt_x2 = model_gp.addMVar(na_ann+1, lb=0, name="sqrt_x2")

    u_min = -3.79
    u_max = 3.91

    y_min = -1.34
    y_max = 5.06

    x_min = -6
    x_max = 6
    
    y_sys = model_gp.addMVar(na_ann+1, lb=y_min, ub=y_max)
    u_past = model_gp.addMVar(na_ann+1, lb=u_min, ub=u_max)

    # Define variables
    for w in range(window):
        y_k[w] = model_gp.addMVar(ny_sys, lb=y_min, ub=y_max, name=f"y_k_{w}")

        h_d[w] = {}
        for j in range(n_layers):
            h_d[w][j] = model_gp.addMVar(n_neurons, lb=0, name=f"h_d{j+1}_{w}")

        x_k[w] = model_gp.addMVar(nx_ann, lb=x_min, ub=x_max, name=f"x_hat_{w}")

        h_e[w] = {}
        for j in range(n_layers):
            h_e[w][j] = model_gp.addMVar(n_neurons, lb=0, name=f"h_e{j+1}_{w}")

        # --- Encoder propagation ---
        # Build information vector
        I_vec = gp.hstack([y_sys[:-1], u_past[1:]])
        
        s["e"] = {}
        delta["e"] = {}

        # Encoder layers
        for j in range(0, n_layers):
            W = encoder_weights[f"layer_{j}"]["W"]
            b = encoder_weights[f"layer_{j}"]["b"]

            s["e"][j] = model_gp.addMVar(W.shape[1], lb=0) # s >= 0
            delta["e"][j] = model_gp.addMVar(W.shape[1], vtype=GRB.BINARY)
            M = 100

            model_gp.addConstr(h_e[w][j] <= M * delta["e"][j])
            model_gp.addConstr(h_e[w][j] >= 0)
            model_gp.addConstr(s["e"][j] >= 0)
            model_gp.addConstr(s["e"][j] <= M * (1 - delta["e"][j]))

            if j == 0:
                for i in range(6):
                    model_gp.addConstr(h_e[w][j][i] - s["e"][j][i] >= (W.T @ I_vec + b)[i])
                    model_gp.addConstr(h_e[w][j][i] - s["e"][j][i] <= (W.T @ I_vec + b)[i])

            else:
                model_gp.addConstr(h_e[w][j] - s["e"][j] >= W.T @ h_e[w][j-1] + b)
                model_gp.addConstr(h_e[w][j] - s["e"][j] <= W.T @ h_e[w][j-1] + b)

        # Latent state constraint
        W = encoder_weights[f"layer_{n_layers}"]["W"]
        b = encoder_weights[f"layer_{n_layers}"]["b"]
        model_gp.addConstr(x_k[w] == W.T @ h_e[w][n_layers-1] + b)

        # Decoder network
        s["d"] = {}
        delta["d"] = {}

        for j in range(0, n_layers):
            W = decoder_weights[f"layer_{j}"]["W"]
            b = decoder_weights[f"layer_{j}"]["b"]

            s["d"][j] = model_gp.addMVar(W.shape[1], lb=0)                  # s >= 0
            delta["d"][j] = model_gp.addMVar(W.shape[1], vtype=GRB.BINARY)  # delta \in {0, 1}          
            M = 100

            model_gp.addConstr(h_d[w][j] <= M * delta["d"][j])
            model_gp.addConstr(h_d[w][j] >= 0)
            model_gp.addConstr(s["d"][j] >= 0)
            model_gp.addConstr(s["d"][j] <= M * (1 - delta["d"][j]))
            if j == 0:
                for i in range(6):
                    model_gp.addConstr(h_d[w][j][i] - s["d"][j][i] >= (W.T @ x_k[w] + b)[i])
                    model_gp.addConstr(h_d[w][j][i] - s["d"][j][i] <= (W.T @ x_k[w] + b)[i])
            else:
                model_gp.addConstr(h_d[w][j] - s["d"][j] >= W.T @ h_d[w][j-1] + b)
                model_gp.addConstr(h_d[w][j] - s["d"][j] <= W.T @ h_d[w][j-1] + b)

        # EQUALITY CONSTRAINT ON EXPECTED OUTPUT
        W = decoder_weights[f"layer_{n_layers}"]["W"][:, -1:]
        b = decoder_weights[f"layer_{n_layers}"]["b"][-1:]
        model_gp.addConstr(y_k[w] == W.T @ h_d[w][n_layers-1] + b)

        # CONSTRAINTS ON NONLINEAR SYSTEM
        for i in range(0, na_ann + 1):
            if i == 0:
                model_gp.addConstr(y_sys[i] == 0)
            else:
                model_gp.addGenConstrPow(x1[i], sqrt_x1[i], 0.5)
                model_gp.addGenConstrPow(x2[i], sqrt_x2[i], 0.5)

                model_gp.addConstr(inflow[i] == 0.4 * u_past[i])

                model_gp.addConstr(x1[i] == x1[i] - 0.5 * sqrt_x1[i] + inflow[i])
                model_gp.addConstr(x2[i] == x2[i] + 0.2 * sqrt_x1[i] - 0.3 * sqrt_x2[i])
                model_gp.addConstr(y_sys[i] == x2[i])


    # --- Objective: Maximize squared error ---
    obj_expr = (y_k[0][0] - y_sys[-1]) * (y_k[0][0] - y_sys[-1])

    model_gp.setObjective(obj_expr, GRB.MAXIMIZE)
    start = time.time()
    model_gp.optimize()
    runtime = time.time() - start 

    time_limit = model_gp.Params.timeLimit
    def softtime(model, where):    
        if where == GRB.Callback.MIP:        
            runtime = model.cbGet(GRB.Callback.RUNTIME)        
            sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)  
            if runtime > 2 * time_limit and sol_count > 0:            
                model.terminate()

    # --- Solve ---
    model_gp.setParam('OutputFlag', 1)
    model_gp.optimize(softtime)

    if model_gp.status in [2,7,8,9,10,11] and model_gp.SolCount > 0:
        y_past = np.array([y_sys[w].X for w in range(na_ann)])
        u_past = np.array([u_past[i].X for i in range(na_ann)]),
        y_fut_sys = np.array([y_sys[w].X for w in range(na_ann, na_ann + window)])
        y_fut_pred = np.array([y_k[w].X for w in range(window)])
        solution = {
            'y_past': y_past,
            'u_past': u_past,
            'y_sys': np.hstack([y_past.squeeze(), y_fut_sys.squeeze()]),
            'y_pred': np.hstack([y_past.squeeze(), y_fut_pred.squeeze()]),
            'x_pred': np.array([x_k[w].X for w in range(window)]),
            
            'h_d': {w: [h_d[w][j].X for j in range(n_layers)] for w in range(window)},
            'h_e': {w: [h_e[w][j].X for j in range(n_layers)] for w in range(window)},
            'runtime':runtime
        }

        return solution
    else:
        return None
