


import time
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from datetime import timedelta
import matplotlib.pyplot as plt

# ==============================================================================
# 1. INITIALIZATION AND SETUP
# ==============================================================================
start_etas_time = time.time()

S = 1.0
sectoyear = 86400.0
Tmax = (maxdate - mindate).total_seconds() / sectoyear

Lats = Datos["Latitude"].to_numpy()
Lons = Datos["Longitude"].to_numpy()
Mags = Datos["Magnitude"].to_numpy()
N = len(Datos)

Times_days = Diferencias / sectoyear
DiferenciasMax_days = np.array([(maxdate - t).total_seconds() for t in Fechas]) / sectoyear

# Gutenberg-Richter b-value and subcriticality constant
bvalue = 1.0 / np.mean(Mags - M0) * np.log10(np.exp(1))
b_ln10 = bvalue * np.log(10)

# Matrix of squared spatial distances (N x N)
dx = Lats[:, None] - Lats[None, :]
dy = Lons[:, None] - Lons[None, :]
Dist_sq = dx**2 + dy**2

dt_days = Times_days[:, None] - Times_days[None, :]
dt_mask = dt_days > 0

# ==============================================================================
# 2. ETAS LIKELIHOOD WITH SUBCRITICALITY CONSTRAINT (\eta < 1)
# ==============================================================================
def compute_nu_matrix(A, alpha, c, p, d):
    K = A * np.exp(alpha * (Mags[None, :] - M0))
    spread = d * np.exp(alpha * (Mags[None, :] - M0))
    F = (1.0 / (2.0 * np.pi * spread)) * np.exp(-Dist_sq / (2.0 * spread))
    G = (p - 1.0) * (c**(p - 1.0)) * np.maximum(dt_days + c, 1e-12)**(-p)
    return np.where(dt_mask, K * F * G, 0.0)

def likelihhod_numpy(theta_5par):
    A, alpha, c, p, d = theta_5par
    
    # Physical constraint for subcritical process (\eta < 1)
    if alpha >= b_ln10 or (A * b_ln10) / (b_ln10 - alpha) >= 1.0:
        return -np.inf

    nu_matrix = compute_nu_matrix(A, alpha, c, p, d)
    nu_ejec = np.sum(nu_matrix, axis=1)
    
    lambda_vals = ubar + nu_ejec
    if np.any(lambda_vals <= 0):
        return -np.inf
    
    log_sum = np.sum(np.log(lambda_vals))
    
    K_i = A * np.exp(alpha * (Mags - M0))
    g_time_integrated = 1.0 - (c / (DiferenciasMax_days + c))**(p - 1.0)
    
    integral_triggered = np.sum(K_i * g_time_integrated)
    integral_background = muintegrated
    
    return log_sum - (integral_background + integral_triggered)

def neg_likelihood_zhuan(theta_5par):
    val = -likelihhod_numpy(theta_5par)
    return val if not np.isnan(val) else 1e12

# Callback to monitor updates
def callback_powell(xk):
    A, alpha, c, p, d = xk
    print(f"   [Paso Powell] A={A:.5f}, alpha={alpha:.5f}, c={c:.5f}, p={p:.5f}, d={d:.5f}")

# ==============================================================================
# 3. EM LOOP WITH POWELL OPTIMIZER (FAST AND EFFECTIVE)
# ==============================================================================
Const_5par = [
    (0.001, 0.9999),  # A
    (0.05, 1.80),     # alpha
    (1e-3, 5.0),      # c
    (1.0001, 3.5),    # p
    (1e-4, 0.1)       # d
]

# Initial seed
A, alpha, c, p, d = 0.118, 1.112, 0.021, 1.363, 0.0048

ubar = np.ones(N) * (N / Tmax)
muintegrated = float(N)
rhojs = np.zeros(N)

# Background smoothing (~38 km)
min_bandwidth = 0.35  
np_eff = 10.0

sorted_idx = np.argsort(Dist_sq, axis=1)

print("\n" + "="*60)
print(" INICIANDO AJUSTE ETAS ZHUANG CON OPTIMIZADOR POWELL")
print("="*60)

for it in range(10):
    print(f"\n--- Iteración EM {it + 1}/10 ---")
    
    # a) Bandwidth d_j according to accumulated background mass
    bg_weights = 1.0 - rhojs
    djs = np.zeros(N)
    for i in range(N):
        cum_bg = np.cumsum(bg_weights[sorted_idx[i]])
        idx_match = np.searchsorted(cum_bg, np_eff)
        if idx_match < N:
            djs[i] = np.sqrt(Dist_sq[i, sorted_idx[i, idx_match]])
        else:
            djs[i] = np.sqrt(Dist_sq[i, sorted_idx[i, -1]])
            
    djs = np.maximum(djs, min_bandwidth)
    
    # b) Update spatial intensity ubar
    weight_background = bg_weights / (2.0 * np.pi * (djs**2))
    kernel_spatial_bg = np.exp(-Dist_sq / (2.0 * (djs[None, :]**2)))
    ubar = np.sum(weight_background[None, :] * kernel_spatial_bg, axis=1) / Tmax
    muintegrated = np.sum(bg_weights)
    
    # c) POWELL OPTIMIZATION (Conjugate direction search without numerical gradients)
    theta_init = (A, alpha, c, p, d)
    
    Opt = minimize(
        fun=neg_likelihood_zhuan,
        x0=theta_init,
        bounds=Const_5par,
        method="Powell",
        callback=callback_powell,
        options={"ftol": 1e-4, "xtol": 1e-4, "maxiter": 200}
    )
    
    A, alpha, c, p, d = Opt["x"]
    print(f" -> Resultado EM {it + 1}: A={A:.5f}, alpha={alpha:.5f}, c={c:.5f}, p={p:.5f}, d={d:.5f}")
    
    # d) Update triggering probabilities \rho_j
    nu_matrix = compute_nu_matrix(A, alpha, c, p, d)
    nu_ejec = np.sum(nu_matrix, axis=1)
    lambda_vals = ubar + nu_ejec
    
    mask = lambda_vals > 0
    rhojs[mask] = np.minimum(nu_ejec[mask] / lambda_vals[mask], 1.0)

elapsed_etas_time = time.time() - start_etas_time

eta_rate = (A * b_ln10) / (b_ln10 - alpha)
print("\n" + "="*60)
print(" RESULTADOS FINALES ETAS ZHUANG (OPTIMIZADOR POWELL)")
print("="*60)
print(f"A     = {A:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"c     = {c:.6f} días")
print(f"p     = {p:.6f}")
print(f"d     = {d:.6f}°")
print(f"Masa de fondo promedio (1 - rho): {np.mean(1 - rhojs):.4f}")
print(f"Tasa de ramificación (eta):       {eta_rate:.4f}")
print(f"Tiempo total de ejecución:        {elapsed_etas_time:.2f} segundos")
print("="*60)

# ==============================================================================
# GENERATION OF THE COMPLETE ETAS DENSITY MAP
# ==============================================================================

lims=np.arange(0,0.03,0.001)

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

# 1. Base map/states layer (geopandas)
if 'shape' in globals() and HAS_GEOPANDAS:
    shape.plot(ax=ax, edgecolor='black', facecolor='whitesmoke', alpha=1, zorder=1)
else:
    ax.set_facecolor('#f7f7f7')

# 2. Continuous surface of background spatial density uxy_grid
contour = ax.contourf(
    Xg, Yg, uxy_grid, 
    levels=lims, 
    cmap='viridis', 
    alpha=0.8, 
    zorder=2
)
cbar = fig.colorbar(contour, ax=ax, shrink=0.82)
cbar.set_label(r'events / ($\text{deg}^2 \cdot \text{day}$)', fontsize=15)

# 3. Cocos Trench
if 'trench_x' in globals() and 'trench_y' in globals():
    ax.plot(trench_x, trench_y, color='deeppink', linewidth=2.2, label='Trinchera de Cocos', zorder=5)

# 4. Observed earthquakes (Data)
ax.scatter(
    Lons, Lats,
    s=12, color='navy', alpha=0.65, edgecolors='none',
    label=f'Sismos $M \ge {M0}$ ($N={N}$)', zorder=6
)

# Map style and limits
ax.set_xlim(xlim, Xlim)
ax.set_ylim(ylim, Ylim)
ax.set_xlabel('Longitud (°)', fontsize=15)
ax.set_ylabel('Latitud (°)', fontsize=15)
# ax.set_title('Mapa de Densidad Espacial de Fondo - ETAS (Zhuang Powell)', fontsize=12, fontweight='bold')
# ax.legend(loc='lower left', framealpha=0.92)
ax.grid(True, linestyle=':', alpha=0.5, color='gray', zorder=3)

plt.tight_layout()

# Save figure to disk
plt.savefig('Mapa_Densidad_ETAS_Zhuang.png', bbox_inches='tight')
plt.show()















