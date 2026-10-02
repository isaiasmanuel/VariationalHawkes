import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.linalg as sla
import time


# ==============================================================================
FUNCTIONS 
# ==============================================================================

def logLike_opt(Muestra, m, lambdas, sigma2, T):
    diff = Muestra[:, None, :] - m[None, :, :]
    sqdist = np.sum(diff**2, axis=2)
    coef = (2 * np.pi * sigma2)**(-Muestra.shape[1] / 2)
    G = coef * np.exp(-sqdist / (2 * sigma2))
    results = (G * np.exp(lambdas)).sum(axis=1)
    return np.sum(np.log(results)) - (np.sum(np.exp(lambdas)) * T)

def grad_logweights_and_means(Muestra, m,Kfr, Kw, lambdas, sigma2, T, Chi, Omega, diff, sqnorm, diff_m):
    lam = np.exp(lambdas)
    n, d = Muestra.shape

    LChi = lam * Chi  
    rho = np.sum(LChi, axis=1) 

    # ---- LOGWEIGHTS GRADIENT ----
    term1_w = Kfr * lam * np.sum(Chi, axis=0)
    term2_w = Kfr * T * lam * (Omega @ lam)
    
    # optimize=True evaluates the best contraction order for speed
    E_diff = np.einsum('nr,nrd->nd', LChi, diff, optimize=True) 
    part_B = np.einsum('nid,nd->ni', diff, E_diff, optimize=True) 
    
    geom_w = (sqnorm - d * sigma2) * rho[:, None] + part_B 
    term3_w = Kw * np.sum((LChi / (sigma2**4 * rho[:, None])) * geom_w, axis=0)

    g_weights = term1_w - term2_w - term3_w

    # ---- MEANS GRADIENT ----
    g_means = Kfr * (lam / sigma2)[:, None] * np.sum(Chi[:, :, None] * diff, axis=0)
    lam_prod = lam[:, None] * lam[None, :]  
    g_means += Kfr * (T / (2 * sigma2)) * np.sum(lam_prod[:, :, None] * diff_m * Omega[:, :, None], axis=1)

    sum_term3 = (sqnorm - (d + 2) * sigma2) * rho[:, None] + part_B 
    geom_m = diff * sum_term3[:, :, None] - sigma2 * E_diff[:, None, :] 
    
    LChi_scaled = LChi / (sigma2**6 * rho[:, None]) 
    accum_m = np.einsum('ni,nid->id', LChi_scaled, geom_m, optimize=True) 
    
    g_means -= Kw * accum_m

    return g_weights, g_means

def build_Gram_matrix(m, lambdas, sigma2, diff_m, Omega_base, coef_omega, const):
    k, d = m.shape
    exp_lam_sum = np.exp(lambdas[:, None] + lambdas[None, :])
    
    # Block 1: Lambda-Lambda
    G_LL = exp_lam_sum * coef_omega * Omega_base
    
    # Block 2: Lambda-Mean
    G_LM_3d = - (exp_lam_sum * const / 2)[:, :, None] * diff_m * Omega_base[:, :, None]
    np.einsum('iij->ij', G_LM_3d)[...] = 0 
    
    G_LM = G_LM_3d.reshape(k, k * d)
    G_ML = G_LM.T  
    
    # Block 3: Mean-Mean
    term_diff = diff_m[:, :, :, None] * diff_m[:, :, None, :] / (4 * sigma2)
    delta = np.eye(d)[None, None, :, :]
    
    G_MM_4d = const * exp_lam_sum[:, :, None, None] * Omega_base[:, :, None, None] * (delta / 2.0 - term_diff)
    
    diag_vals = (const / 2) * np.exp(2 * lambdas)[:, None, None] * np.eye(d)[None, :, :]
    idx = np.arange(k)
    G_MM_4d[idx, idx, :, :] = diag_vals
    
    G_MM = G_MM_4d.transpose(0, 2, 1, 3).reshape(k * d, k * d)
    
    G = np.block([
        [G_LL, G_LM],
        [G_ML, G_MM]
    ])
    return G

# ==============================================================================
# OPTIMIZATION FUNCTION
# ==============================================================================

def run_hawkes_optimization(
    Datos,
    mindate,
    maxdate,
    k=100,
    sigma2=1,
    Kfr=1,
    Kw=1,
    e=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    tolerancia_global=1e-2,
    PasosGrafica=100,
    seed=1,
    show_intermediate_plots=False,
    fig_filename="last_figure.png",
    params_filename="hawkes_params.npz"
):
    # ==============================================================================
    # INITIAL CONFIGURATION
    # ==============================================================================

    Muestra = Datos[["Longitude", "Latitude"]].to_numpy() 
    n_muestra, d_dim = Muestra.shape

    fechas_df = pd.to_datetime(Datos[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    T = (maxdate - mindate).total_seconds() / (365 * 24 * 60 * 60)

    np.random.seed(seed)
    tadam = 0

    m = np.array(Datos.loc[np.random.choice(len(Datos), size=k)][["Longitude", "Latitude"]])  
    lambdas = np.log(np.ones(len(m)) / len(m) * len(Datos) / T)

    ejex = np.linspace(np.min(Datos.iloc[:,0])-1, np.max(Datos.iloc[:,0])+1, 20)
    ejey = np.linspace(np.min(Datos.iloc[:,1])-1, np.max(Datos.iloc[:,1])+1, 20)
    ejeX, ejeY = np.meshgrid(ejex, ejey)

    veros = []

    # Precalculated constants (do not change)
    coef_X_m = (2 * np.pi * sigma2)**(-d_dim / 2)
    coef_omega = (4 * np.pi * sigma2)**(-d_dim / 2)
    const_gram = (2*np.pi)**(-d_dim) * np.pi**(d_dim/2) * (sigma2)**(-(d_dim+2)/2)

    ######### Stopping criterion
    veros_previo = -np.inf

    # ==============================================================================
    # OPTIMIZATION LOOP
    # ==============================================================================

    for CoordinateSteps in range(50):
        veros_actual = logLike_opt(Muestra, m, lambdas, sigma2, T)
        cambio_veros = np.abs(veros_actual - veros_previo)
        
        print(f"--> Fin del Paso {CoordinateSteps} | Log-Like: {veros_actual:.4f} | Mejora: {cambio_veros:.6f}")
        
        if cambio_veros < tolerancia_global:
            print(f"Convergencia global alcanzada en el paso {CoordinateSteps}. El algoritmo ha finalizado.")
            break
            
        veros_previo = veros_actual
        
        # 1. Vectorized cleaning
        diff = m[:, None, :] - m[None, :, :]
        matriz_distancias = np.sqrt(np.sum(diff**2, axis=2))
        tol = 0.001
        
        pares_cercanos = np.triu(matriz_distancias < tol, k=1)
        indices_a_eliminar = np.where(pares_cercanos)[1]
        indaux = np.setdiff1d(np.arange(len(m)), indices_a_eliminar)
        
        maux = m[indaux]
        lambdasaux = np.zeros(len(indaux))
        
        for idx, i in enumerate(indaux):
            cercanos = np.where(matriz_distancias[i] < tol)[0]
            lambdasaux[idx] = np.log(np.sum(np.exp(lambdas[cercanos])))
            
        m = np.copy(maux)
        lambdas = np.copy(lambdasaux)

        # 2. Elimination of tiny mass
        mascara = lambdas > -10
        m = np.copy(m[mascara, :])
        lambdas_validos = np.exp(lambdas[mascara])
        masa_perdida = np.sum(np.exp(lambdas[~mascara]))
        
        if len(lambdas_validos) > 0:
            lambdas = np.log(lambdas_validos + masa_perdida * (lambdas_validos / np.sum(lambdas_validos)))
            
        m_alpha = np.zeros(len(m))
        v_alpha = np.zeros(len(m))
        m_m = np.zeros(len(m) * 2)
        v_m = np.zeros(len(m) * 2)

        for rep in range(1000):
            # -- GEOMETRIC CACHE (Calculated once per Adam iteration) --
            diff_X_m = Muestra[:, None, :] - m[None, :, :]
            sqdist_X_m = np.sum(diff_X_m**2, axis=2)
            Chi = coef_X_m * np.exp(-sqdist_X_m / (2 * sigma2))
            
            diff_m = m[:, None, :] - m[None, :, :]
            sqdist_m = np.sum(diff_m**2, axis=2)
            Omega_base = np.exp(-sqdist_m / (4 * sigma2))
            Omega = coef_omega * Omega_base
            # -------------------------------------------------------------

            # Passing stored variables
            g_lambda, g_m = grad_logweights_and_means(Muestra, m, Kfr, Kw, lambdas, sigma2, T, Chi, Omega, diff_X_m, sqdist_X_m, diff_m)
            
            c = np.empty((g_m[:,0].size * 2,), dtype=g_m.dtype)
            c[0::2] = g_m[:,0]
            c[1::2] = g_m[:,1]
            gfinal = np.concatenate((g_lambda, c))
            
            norma_gradiente = np.max(np.abs(gfinal))
            if norma_gradiente < 1e-4:
                print(f"Convergencia alcanzada en el paso {rep}")
                break
            
            G = build_Gram_matrix(m, lambdas, sigma2, diff_m, Omega_base, coef_omega, const_gram)
            
            epsilon_reg = 1e-4 * (np.trace(G) / G.shape[0])
            G_estabilizada = G + epsilon_reg * np.eye(G.shape[0])
            
            thetadot = sla.solve(G_estabilizada, gfinal, assume_a='pos', overwrite_a=True, overwrite_b=True)
            tadam += 1
            
            td_alpha = thetadot[:len(m)]
            td_m = thetadot[len(m):]
            
            m_alpha = beta1 * m_alpha + (1 - beta1) * td_alpha
            v_alpha = beta2 * v_alpha + (1 - beta2) * (td_alpha ** 2)
            m_alpha_hat = m_alpha / (1 - beta1 ** tadam)
            v_alpha_hat = v_alpha / (1 - beta2 ** tadam)

            m_m = beta1 * m_m + (1 - beta1) * td_m
            v_m = beta2 * v_m + (1 - beta2) * (td_m ** 2)
            m_m_hat = m_m / (1 - beta1 ** tadam)
            v_m_hat = v_m / (1 - beta2 ** tadam)          
            
            step_alpha = e * m_alpha_hat / (np.sqrt(v_alpha_hat) + epsilon)
            step_m = e * m_m_hat / (np.sqrt(v_m_hat) + epsilon)
            
            lambdas += step_alpha
            m[:, 0] += step_m[0::2]
            m[:, 1] += step_m[1::2]
            
            if rep % PasosGrafica == 0:
                veros.append(logLike_opt(Muestra, m, lambdas, sigma2, T)) 
                
                if show_intermediate_plots:
                    grid_coords = np.column_stack((ejeX.ravel(), ejeY.ravel()))
                    grid_diff = grid_coords[:, None, :] - m[None, :, :]
                    grid_sqdist = np.sum(grid_diff**2, axis=2)
                    grid_Chi = coef_X_m * np.exp(-grid_sqdist / (2 * sigma2))
                    ejeZ = (grid_Chi * np.exp(lambdas)).sum(axis=1).reshape(ejeX.shape)
                    
                    fig, ax = plt.subplots(1, 2, figsize=(12,5))
                    ax[0].plot(veros)
                    ax[0].set_title("Log-Likelihood")
                    
                    cont = ax[1].contourf(ejeX, ejeY, ejeZ, levels=20)
                    ax[1].scatter(Muestra[:,0], Muestra[:,1], s=10)
                    ax[1].scatter(m[:,0], m[:,1], c="red")
                    ax[1].set_title(f"k = {len(m)}")
                    fig.colorbar(cont, ax=ax[1])
                    
                    plt.tight_layout()
                    plt.show()

    # Generate and save only the final figure
    grid_coords = np.column_stack((ejeX.ravel(), ejeY.ravel()))
    grid_diff = grid_coords[:, None, :] - m[None, :, :]
    grid_sqdist = np.sum(grid_diff**2, axis=2)
    grid_Chi = coef_X_m * np.exp(-grid_sqdist / (2 * sigma2))
    ejeZ = (grid_Chi * np.exp(lambdas)).sum(axis=1).reshape(ejeX.shape)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(veros)
    ax[0].set_title("Log-Likelihood")
    
    cont = ax[1].contourf(ejeX, ejeY, ejeZ, levels=20)
    ax[1].scatter(Muestra[:,0], Muestra[:,1], s=10)
    ax[1].scatter(m[:,0], m[:,1], c="red")
    ax[1].set_title(f"k = {len(m)}")
    fig.colorbar(cont, ax=ax[1])
    
    plt.tight_layout()
    if fig_filename:
        fig.savefig('./CorridaPoisson/'+fig_filename)
    plt.show()
    plt.close(fig)

    # Save the values of m and lambdas to a numpy file
    if params_filename:
        np.savez('./CorridaPoisson/'+params_filename, m=m, lambdas=lambdas)

    return m, lambdas


start = time.time()

delta=0.25
for sigma2 in np.arange(0.1,5+delta,0.25):
    
    run_hawkes_optimization(
        Datos,
        mindate,
        maxdate,
        k=100,
        sigma2=sigma2,
        Kfr=1,
        Kw=1,
        e=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        tolerancia_global=1e-2,
        PasosGrafica=100,
        seed=1,
        show_intermediate_plots=False,
        fig_filename="Optimizacion"+str(sigma2)+".png",
        params_filename="Params_Poisson"+str(sigma2)+".npz"
    )



def compute_aic_bic_from_file(params_filename, Datos, mindate, maxdate, sigma2):
    # 1. Load saved parameters (m and lambdas)
    params = np.load(params_filename)
    m = params['m']
    lambdas = params['lambdas']
    
    # 2. Reconstruct spatial data and time horizon T
    Muestra = Datos[["Longitude", "Latitude"]].to_numpy()
    N = len(Muestra)  # Número total de observaciones/eventos
    T = (maxdate - mindate).total_seconds() / (365 * 24 * 60 * 60)
    
    # 3. Calculate Log-Likelihood
    log_likelihood = logLike_opt(Muestra, m, lambdas, sigma2, T)
    
    # 4. Count estimated parameters: k components * (2 coordinates + 1 log-weight)
    k_components, d_dim = m.shape
    num_params = k_components * (d_dim + 1)
    
    # 5. Information Criteria
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(N) - 2 * log_likelihood
    
    return aic, bic, log_likelihood, k_components, N

# ==============================================================================
# AIC EVALUATION WITH EXACT NAMES
# ==============================================================================



delta = 0.25
results = []

for sigma2 in np.arange(0.1, 5 + delta, 0.25):
    # Exact string as it was saved
    filename = "./CorridaPoisson/"+"Params_Poisson"+str(sigma2) + ".npz"
    
    if os.path.exists(filename):
        aic, bic, log_like, k_comp, N_obs = compute_aic_bic_from_file(
            filename, Datos, mindate, maxdate, sigma2
        )
        results.append({
            "sigma2": sigma2,
            "AIC": aic,
            "BIC": bic,
            "LogLikelihood": log_like,
            "Components_k": k_comp,
            "NumParameters": k_comp * 3,
            "N_obs": N_obs
        })
    else:
        print(f"Archivo no encontrado: {filename}")

df_results = pd.DataFrame(results)

if not df_results.empty:
    print(df_results.to_string(index=False))
    
    best_aic = df_results.loc[df_results['AIC'].idxmin()]
    best_bic = df_results.loc[df_results['BIC'].idxmin()]
    
    print("\n" + "="*50)
    print("Modelo óptimo según AIC:")
    print(f"  sigma2: {best_aic['sigma2']} | k: {best_aic['Components_k']} | AIC: {best_aic['AIC']:.4f}")
    
    print("\nModelo óptimo según BIC:")
    print(f"  sigma2: {best_bic['sigma2']} | k: {best_bic['Components_k']} | BIC: {best_bic['BIC']:.4f}")
    print("="*50)
else:
    print("\nNo se encontraron archivos .npz para evaluar.")



end = time.time()
print(end - start)








###################
###################
###################
###################
###################


x = np.linspace(-5,10,20)
y = np.linspace(-5,10,20)
Xg,Yg = np.meshgrid(x,y)
pdf0 = np.zeros(Xg.shape)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Obs=np.array((Xg[i,j], Yg[i,j],t))
        pdf0[i,j] = intensity(Obs)    
levels = np.linspace(pdf0.min(), 4.5, 10)
plt.plot()
contour=plt.contour(Xg, Yg, pdf0, cmap='viridis',levels=levels )
plt.colorbar(contour)
# plt.title( "t="+str(np.round(t,2)))
plt.show()

###################
###################
###################
###################
###################

# 1. Load parameters of the optimal model according to AIC
best_row = df_results.loc[df_results['AIC'].idxmin()]
best_sigma2 = best_row['sigma2']

filename = f"./CorridaPoisson/Params_Poisson{best_sigma2}.npz"
if not os.path.exists(filename):
    filename = f"Params_Poisson{best_sigma2}.npz"

params = np.load(filename)
m = params['m']
lambdas = params['lambdas']

# 2. Definition of the grid and time
t = 0
x = np.linspace(-5, 10, 20)
y = np.linspace(-5, 10, 20)
Xg, Yg = np.meshgrid(x, y)


plt.scatter(m[:,0],m[:,1], c="red")

# 3. Vectorized calculation (identical to run_hawkes_optimization)
coef_X_m = (2 * np.pi * best_sigma2)**(-1.0)
grid_coords = np.column_stack((Xg.ravel(), Yg.ravel()))
grid_diff = grid_coords[:, None, :] - m[None, :, :]
grid_sqdist = np.sum(grid_diff**2, axis=2)
grid_Chi = coef_X_m * np.exp(-grid_sqdist / (2 * best_sigma2))
pdf0 = (grid_Chi * np.exp(lambdas)).sum(axis=1).reshape(Xg.shape)

# 4. Plotting
plt.plot()
contour = plt.contour(Xg, Yg, pdf0, cmap='viridis', levels=levels)
plt.colorbar(contour)
# plt.title("Mejor modelo")
plt.show()