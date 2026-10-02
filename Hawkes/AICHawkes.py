

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.linalg as sla
from scipy.optimize import minimize
import os
import time


# ==============================================================================
# 1. MATHEMATICAL FUNCTIONS AND VECTORIZED HAWKES PROCESS
# ==============================================================================

def gaussian_pdf(X, MU, sigma2):
    diff = X[:, None, :] - MU[None, :, :]
    sqdist = np.sum(diff**2, axis=2)
    D = X.shape[1]
    coef = (2 * np.pi * sigma2)**(-D / 2)
    return coef * np.exp(-sqdist / (2 * sigma2))

def Aproximante(z, m, lambdas, sigma2):
    z = np.atleast_2d(z)
    G = gaussian_pdf(z, m, sigma2)
    return (G * np.exp(lambdas)).sum(axis=1)

def logLike_opt(Muestra, m, lambdas, sigma2, T):
    results = Aproximante(Muestra, m, lambdas, sigma2)
    return np.sum(np.log(results)) - (np.sum(np.exp(lambdas)) * T)

# --- VECTORIZED HAWKES KERNEL ---
def compute_nu_vectorized(A, alpha, cparam, p, d_val, M0, Times, Lats, Lons, Mags):
    dt_seconds = Times[:, None] - Times[None, :]
    dt_days = dt_seconds / 86400.0  
    dt_mask = dt_seconds > 0  # Strict temporal precedence mask

    dx = Lons[:, None] - Lons[None, :]
    dy = Lats[:, None] - Lats[None, :]
    sqdist = dx**2 + dy**2

    K = A * np.exp(alpha * (Mags[None, :] - M0))
    spread = d_val * np.exp(alpha * (Mags[None, :] - M0))
    F = (1 / (2 * np.pi * spread)) * np.exp(-sqdist / (2 * spread))
    
    G_time = (p - 1) * (cparam**(p - 1)) * ((dt_days + cparam)**(-p))
    
    nu_matrix = K * F * G_time
    # Apply temporal restrictions and sum the influence of j (columns) on i (rows)
    return np.sum(np.where(dt_mask, nu_matrix, 0.0), axis=1)

def likelihood_opt(thetalike, bvalue, M0, Muestra, m, lambdas, sigma2, Times, Lats, Lons, Mags):
    A, alpha, cparam, p, d_val = thetalike

    # Penalty for Hawkes instability
    if (A * bvalue * np.log(10)) / (bvalue * np.log(10) - alpha) >= 1:
        return -np.inf 

    nu_ejec = compute_nu_vectorized(A, alpha, cparam, p, d_val, M0, Times, Lats, Lons, Mags)
    muxy = Aproximante(Muestra, m, lambdas, sigma2)
    
    intensidad = np.sum(np.log(muxy + nu_ejec))
    intensidad -= np.sum(A * np.exp(alpha * (Mags - M0)))

    global thetaIteracion
    thetaIteracion = np.copy(thetalike)
    # print(f"Likelihood: {intensidad:.4f} | Alpha: {alpha:.4f}")
    return intensidad

def likelihood_full(thetalike, bvalue, M0, Muestra, m, lambdas, sigma2, Times, Lats, Lons, Mags, T):
    A, alpha, cparam, p, d_val = thetalike

    nu_ejec = compute_nu_vectorized(A, alpha, cparam, p, d_val, M0, Times, Lats, Lons, Mags)
    muxy = Aproximante(Muestra, m, lambdas, sigma2)
    
    intensidad = np.sum(np.log(muxy + nu_ejec))
    intensidad -= (np.sum(np.exp(lambdas)) * T)
    intensidad -= np.sum(A * np.exp(alpha * (Mags - M0)))

    return intensidad

# ==============================================================================
# 2. GRADIENTS AND COLLAPSED GRAM MATRIX FOR HAWKES
# ==============================================================================

def grad_logweights_and_means_hawkes(Muestra, m, lambdas, sigma2, T, nu_ejecutados, Chi, Omega, diff, sqnorm, diff_m, Kfr, Kw):
    lam = np.exp(lambdas)
    n, d = Muestra.shape

    LChi = lam * Chi
    rho = np.sum(LChi, axis=1)
    rho_hawkes = rho + nu_ejecutados

    # ---- LOGWEIGHTS GRADIENT ----
    term1_w = Kfr * lam * np.sum((Chi * rho[:, None]) / rho_hawkes[:, None], axis=0)
    term2_w = Kfr * T * lam * (Omega @ lam)

    E_diff = np.einsum('nr,nrd->nd', LChi, diff, optimize=True)
    part_B = np.einsum('nid,nd->ni', diff, E_diff, optimize=True)

    geom_w = (sqnorm - d * sigma2) * rho[:, None] + part_B
    term3_w = Kw * np.sum((LChi / (sigma2**4 * rho_hawkes[:, None])) * geom_w, axis=0)

    g_weights = term1_w - term2_w - term3_w

    # ---- MEANS GRADIENT ----
    factor_term1_m = (Chi * rho[:, None]) / rho_hawkes[:, None]
    g_means = Kfr * (lam / sigma2)[:, None] * np.sum(factor_term1_m[:, :, None] * diff, axis=0)

    lam_prod = lam[:, None] * lam[None, :]
    g_means += Kfr * (T / (2 * sigma2)) * np.sum(lam_prod[:, :, None] * diff_m * Omega[:, :, None], axis=1)

    sum_term3 = (sqnorm - (d + 2) * sigma2) * rho[:, None] + part_B
    geom_m = diff * sum_term3[:, :, None] - sigma2 * E_diff[:, None, :]

    LChi_scaled = LChi / (sigma2**6 * rho_hawkes[:, None])
    accum_m = np.einsum('ni,nid->id', LChi_scaled, geom_m, optimize=True)

    g_means -= Kw * accum_m

    return g_weights, g_means

def build_Gram_matrix(m, lambdas, sigma2, diff_m, Omega_base, coef_omega, const):
    k, d = m.shape
    exp_lam_sum = np.exp(lambdas[:, None] + lambdas[None, :])
    
    G_LL = exp_lam_sum * coef_omega * Omega_base
    
    G_LM_3d = - (exp_lam_sum * const / 2)[:, :, None] * diff_m * Omega_base[:, :, None]
    np.einsum('iij->ij', G_LM_3d)[...] = 0 
    G_LM = G_LM_3d.reshape(k, k * d)
    G_ML = G_LM.T  
    
    term_diff = diff_m[:, :, :, None] * diff_m[:, :, None, :] / (4 * sigma2)
    delta = np.eye(d)[None, None, :, :]
    
    G_MM_4d = const * exp_lam_sum[:, :, None, None] * Omega_base[:, :, None, None] * (delta / 2.0 - term_diff)
    
    diag_vals = (const / 2) * np.exp(2 * lambdas)[:, None, None] * np.eye(d)[None, :, :]
    idx = np.arange(k)
    G_MM_4d[idx, idx, :, :] = diag_vals
    
    G_MM = G_MM_4d.transpose(0, 2, 1, 3).reshape(k * d, k * d)
    
    G = np.block([[G_LL, G_LM], [G_ML, G_MM]])
    return G

# ==============================================================================
# 3. HAWKES OPTIMIZATION FUNCTION
# ==============================================================================

def run_hawkes_optimization(
    Datos,
    mindate,
    maxdate,
    bvalue,
    k=100,
    sigma2=0.5,
    Kfr=2,
    Kw=0.1,
    IteracionesGrad=1000,
    e=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    PasosGrafica=100,
    seed=1,
    show_intermediate_plots=False,
    fig_filename="last_figure.png",
    params_filename="hawkes_params.npz"
):
    Muestra = Datos[["Longitude", "Latitude"]].to_numpy()
    n_muestra, d_dim = Muestra.shape

    fechas_df = pd.to_datetime(Datos[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    Times = (fechas_df - mindate).dt.total_seconds().to_numpy()
    Lats = Muestra[:, 1]
    Lons = Muestra[:, 0]
    Mags = Datos["Magnitude"].to_numpy()
    T = (maxdate - mindate).total_seconds() / (24 * 60 * 60) # In days

    np.random.seed(seed)
    tadam = 0

    m = np.array(Datos.loc[np.random.choice(len(Datos), size=k)][["Longitude", "Latitude"]])  
    lambdas = np.log(np.ones(len(m)) / len(m) * len(Datos) / T)

    ejex = np.linspace(np.min(Muestra[:,0])-1, np.max(Muestra[:,0])+1, 20)
    ejey = np.linspace(np.min(Muestra[:,1])-1, np.max(Muestra[:,1])+1, 20)
    ejeX, ejeY = np.meshgrid(ejex, ejey)

    veros = []

    # Precalculated constants (do not change)
    coef_X_m = (2 * np.pi * sigma2)**(-d_dim / 2)
    coef_omega = (4 * np.pi * sigma2)**(-d_dim / 2)
    const_gram = (2*np.pi)**(-d_dim) * np.pi**(d_dim/2) * (sigma2)**(-(d_dim+2)/2)

    # Initial self-excitation parameters
    cparam, p = 0.021, 1.363
    A, alpha, M0 = 0.118, 1.112, 4.3
    d_val = 0.0048

    nu_ejecutados = compute_nu_vectorized(A, alpha, cparam, p, d_val, M0, Times, Lats, Lons, Mags)

    # ==============================================================================
    # MAIN OPTIMIZATION LOOP
    # ==============================================================================

    for CoordinateSteps in range(50):
        print(f"Centros iniciales paso {CoordinateSteps}: {len(m)}")
        
        diff_c = m[:, None, :] - m[None, :, :]
        matriz_distancias = np.sqrt(np.sum(diff_c**2, axis=2))
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

        for rep in range(IteracionesGrad):
            diff_X_m = Muestra[:, None, :] - m[None, :, :]
            sqdist_X_m = np.sum(diff_X_m**2, axis=2)
            Chi = coef_X_m * np.exp(-sqdist_X_m / (2 * sigma2))
            
            diff_m = m[:, None, :] - m[None, :, :]
            sqdist_m = np.sum(diff_m**2, axis=2)
            Omega_base = np.exp(-sqdist_m / (4 * sigma2))
            Omega = coef_omega * Omega_base

            g_lambda, g_m = grad_logweights_and_means_hawkes(Muestra, m, lambdas, sigma2, T, nu_ejecutados, Chi, Omega, diff_X_m, sqdist_X_m, diff_m, Kfr, Kw)
            
            c = np.empty((g_m[:,0].size * 2,), dtype=g_m.dtype)
            c[0::2] = g_m[:,0]
            c[1::2] = g_m[:,1]
            gfinal = np.concatenate((g_lambda, c))
            
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
                    ejeZ = Aproximante(grid_coords, m, lambdas, sigma2).reshape(ejeX.shape)
                    
                    fig, ax = plt.subplots(1, 2, figsize=(12,5))
                    ax[0].plot(veros)
                    ax[0].set_title("Log-Likelihood Background")
                    
                    cont = ax[1].contourf(ejeX, ejeY, ejeZ, levels=20)
                    ax[1].scatter(Muestra[:,0], Muestra[:,1], s=10)
                    ax[1].scatter(m[:,0], m[:,1], c="red")
                    ax[1].set_title(f"k = {len(m)}")
                    fig.colorbar(cont, ax=ax[1])
                    
                    plt.tight_layout()
                    plt.show()
                    print(f"Masa total: {np.sum(np.exp(lambdas)):.4f}")
                
        Const = [ 
            [10**-5, 1-10**(-10)],
            [0.8, 1.5],
            [10**(-8), 5],
            [1+10**(-10), 2],
            [10**(-10), 1]
        ]
        
        Parametros = A, alpha, cparam, p, d_val
        
        Optimizacion = minimize(
            lambda theta: -likelihood_opt(theta, bvalue, M0, Muestra, m, lambdas, sigma2, Times, Lats, Lons, Mags),
            Parametros, 
            bounds=Const, 
            method="Nelder-Mead",    
            options={"maxiter": 200, "fatol": 1e-2}
        )
        
        A, alpha, cparam, p, d_val = Optimizacion["x"]
        nu_ejecutados = compute_nu_vectorized(A, alpha, cparam, p, d_val, M0, Times, Lats, Lons, Mags)

    # Generate and save only the final figure
    grid_coords = np.column_stack((ejeX.ravel(), ejeY.ravel()))
    ejeZ = Aproximante(grid_coords, m, lambdas, sigma2).reshape(ejeX.shape)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(veros)
    ax[0].set_title("Log-Likelihood Background")
    
    cont = ax[1].contourf(ejeX, ejeY, ejeZ, levels=20)
    ax[1].scatter(Muestra[:,0], Muestra[:,1], s=10)
    ax[1].scatter(m[:,0], m[:,1], c="red")
    ax[1].set_title(f"k = {len(m)}")
    fig.colorbar(cont, ax=ax[1])
    
    plt.tight_layout()
    if fig_filename:
        fig.savefig('./CorridaHawkes/'+fig_filename)
    plt.show()
    plt.close(fig)

    if params_filename:
        np.savez('./CorridaHawkes/'+params_filename, m=m, lambdas=lambdas, theta_hawkes=np.array([A, alpha, cparam, p, d_val]), M0=M0)

    return m, lambdas, (A, alpha, cparam, p, d_val)

# ==============================================================================
# 4. AIC AND BIC CALCULATION FOR HAWKES USING LIKELIHOOD_FULL
# ==============================================================================

def compute_aic_bic_from_file(params_filename, Datos, mindate, maxdate, sigma2, bvalue):
    params = np.load(params_filename)
    m = params['m']
    lambdas = params['lambdas']
    theta_hawkes = params['theta_hawkes']
    M0 = float(params['M0'])
    
    Muestra = Datos[["Longitude", "Latitude"]].to_numpy()
    fechas_df = pd.to_datetime(Datos[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    Times = (fechas_df - mindate).dt.total_seconds().to_numpy()
    Lats = Muestra[:, 1]
    Lons = Muestra[:, 0]
    Mags = Datos["Magnitude"].to_numpy()
    N = len(Muestra)
    T = (maxdate - mindate).total_seconds() / (24 * 60 * 60) # In days
    
    log_likelihood = likelihood_full(theta_hawkes, bvalue, M0, Muestra, m, lambdas, sigma2, Times, Lats, Lons, Mags, T)
    
    k_components, d_dim = m.shape
    num_params = k_components * (d_dim + 1) + len(theta_hawkes)
    
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(N) - 2 * log_likelihood
    
    return aic, bic, log_likelihood, k_components, N

# ==============================================================================
# 5. EXECUTION AND EVALUATION LOOP
# ==============================================================================

delta = 0.1
ranL = 0.1
ranU = 1

start = time.time()

for sigma2 in np.arange(ranL, ranU + delta, delta):
    run_hawkes_optimization(
        Datos,
        mindate,
        maxdate,
        bvalue,
        k=100,
        sigma2=sigma2,
        Kfr=2,
        Kw=0.1,
        IteracionesGrad=1000,
        e=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        PasosGrafica=100,
        seed=1,
        show_intermediate_plots=False,
        fig_filename="Optimizacion_Hawkes_" + str(sigma2) + ".png",
        params_filename="Params_Hawkes_" + str(sigma2) + ".npz"
    )

results = []
for sigma2 in np.arange(ranL, ranU + delta, delta):
    filename = "./CorridaHawkes/" + "Params_Hawkes_" + str(sigma2) + ".npz"
    
    if os.path.exists(filename):
        aic, bic, log_like, k_comp, N_obs = compute_aic_bic_from_file(
            filename, Datos, mindate, maxdate, sigma2, bvalue
        )
        results.append({
            "sigma2": sigma2,
            "AIC": aic,
            "BIC": bic,
            "LogLikelihood": log_like,
            "Components_k": k_comp,
            "NumParameters": k_comp * 3 + 5,
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

best_row = df_results.loc[df_results['AIC'].idxmin()]
best_sigma2 = best_row['sigma2']

filename = f"./CorridaHawkes/Params_Hawkes_{best_sigma2}.npz"
if not os.path.exists(filename):
    filename = f"Params_Hawkes_{best_sigma2}.npz"

params = np.load(filename)
m = params['m']
lambdas = params['lambdas']    

S = 0.5  # Margin around the seismic region
xlim, Xlim = np.min(Datos["Longitude"]) - S, np.max(Datos["Longitude"]) + S
ylim, Ylim = np.min(Datos["Latitude"]) - S, np.max(Datos["Latitude"]) + S

grid_resolution = 200
x = np.linspace(xlim, Xlim, grid_resolution)
y = np.linspace(ylim, Ylim, grid_resolution)
Xg, Yg = np.meshgrid(x, y)
    
grid_coords = np.column_stack((Xg.ravel(), Yg.ravel()))
pdf0 = Aproximante(grid_coords, m, lambdas, best_sigma2).reshape(Xg.shape)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# Filled contour map of the background spatial density
contour = ax.contourf(Xg, Yg, pdf0, levels=30, cmap='YlOrRd', alpha=0.85)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(r'Densidad Espacial de Fondo $\mu(x,y)$ [eventos / ($\text{deg}^2 \cdot \text{día}$)]')

# Superposition of earthquakes and Gaussian centers
ax.scatter(Datos["Longitude"], Datos["Latitude"], s=10, color='navy', alpha=0.5, label='Sismos observados')
ax.scatter(m[:, 0], m[:, 1], s=25, color='cyan', marker='x', linewidths=1.2, label=f'Centros WFR ($k={len(m)}$)')

ax.set_xlim(xlim, Xlim)
ax.set_ylim(ylim, Ylim)
ax.set_xlabel('Longitud (°)', fontsize=11)
ax.set_ylabel('Latitud (°)', fontsize=11)
ax.set_title(f'Mejor Modelo Hawkes según AIC ($\sigma^2 = {best_sigma2}$)', fontsize=12, fontweight='bold')
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

params["theta_hawkes"]
#(A, alpha, cparam, p, d_val)