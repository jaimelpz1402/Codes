import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
import math
# =============================================================================
# PERSONALIZACIÓN DE TEXTOS (leyendas, ejes y títulos)
# =============================================================================
# --- Gráfica 1 (W vs SyW con ratio de áreas gaussianas) ---
TITLE_PART1   = "S and W spectrum (E_beam = 3800 eV) normalized to pure W peak at 2179.3 $\\pm$ 3eV"
XLABEL_PART1  = "Photon energy (eV)"
YLABEL_PART1  = "Counts"
LEGEND_W      = "W {beam:.0f} eV (±{tol})"
LEGEND_SYW    = "S 3800 eV"
LEGEND_FIT_W  = "W Gaussian"
LEGEND_FIT_SYW= "S Gaussian"
LEGEND_MU_W   = "W peak"
LEGEND_MU_SYW = "S peak"

# --- Gráfica 2 (un solo espectro con gaussianas; solo etiquetas de energía) ---
TITLE_PART2   = "Sulfur contamination due to tungsten at E_beam = 3800 eV and gaussian fits with 3 eV window"
XLABEL_PART2  = "Photon energy (eV)"
YLABEL_PART2  = "Counts"
LEGEND_SPEC   = "Spectrum"
LEGEND_FIT    = "Gaussian fit"
LEGEND_MAX    = "Gaussian maxima"

# =============================================================================
# RUTAS / ENTRADAS
# =============================================================================
# --- PRIMERA PARTE (SyW vs W por RATIO de áreas gaussianas; % por pico en memoria) ---
PATH_DATA_A_NPY    = "/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy" # W puro: data[1,:]=E_beam, data[2,:]=E_photon
PATH_DATA_B_NPY    = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run003_2025_01_30.npy" # SyW:   E, counts (2 filas o 2 columnas)
PATH_PEAKS_W_TXT   = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/W3800.txt" # energías (eV) de picos de W (se lee, no se crea)
PATH_PEAKS_SYW_TXT = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/SyW3800.txt" # energías (eV) de picos SyW (se lee, no se crea)

# --- SEGUNDA PARTE (un solo espectro + gaussianas; usa % del mapa en memoria) ---
PATH_SPECTRUM_NPY  = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run002_40pix.npy"
PATH_PEAKS_TXT     = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/COMPARAR3800.txt" # se lee, no se crea
IMG_DIR_2          = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"

# --- Salidas (solo imágenes PNG; no se crean txt/csv) ---
OUTPUT_DIR         = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR_2, exist_ok=True)

# =============================================================================
# PARÁMETROS 1ª PARTE (SyW vs W)
# =============================================================================
TARGET_BEAM_ENERGY_EV = 3800.0
BEAM_TOLERANCE_EV     = 50.0
X_MIN_EV   = 700.0
X_MAX_EV   = 4500.0
NUM_BINS   = 3000

# Normalización base ANTES de ajustar: SIEMPRE 'peak'
NORMALIZE_MODE = "peak"            # fijo
PEAK_CENTER_EV  = 2179.8
PEAK_WINDOW_EV  = 5.0

# Normalización adicional SOLO para el PLOT (no afecta a integrales)
PLOT_NORMALIZE_MODE = "none"       # "none" | "max" | "area"

# Emparejado por proximidad
MATCH_MAX_DELTA_EV = 5.0           # tolerancia mínima absoluta
FWHM_HINT_EV       = 5.0           # pista de FWHM si no mides resolución aquí
PAIR_TOL_FACTOR    = 0.5           # tolerancia efectiva = max(MATCH_MAX_DELTA_EV, PAIR_TOL_FACTOR*FWHM_HINT_EV)

# Ajuste de picos (ventanas locales)
WINDOW_MARGIN_EV = 6.0

# Presentación del % en la graf. 1
DISPLAY_MIN_PERCENT = 0.01

# Estética
SPECTRUM_ALPHA = 0.85
LINEWIDTH = 1.6
DPI_SAVE = 200
COLOR_W = '#1f77b4'
COLOR_S = '#ff7f0e'
COLOR_FIT_W = '#2ca02c'
COLOR_FIT_S = '#d62728'
MARKER_W = dict(marker='o', s=20, color=COLOR_FIT_W, edgecolor='white', linewidth=0.6, zorder=12)
MARKER_S = dict(marker='o', s=22, color=COLOR_FIT_S, edgecolor='white', linewidth=0.6, zorder=12)

# =============================================================================
# PARÁMETROS 2ª PARTE (un espectro)
# =============================================================================
WINDOW_MARGIN_EV_2 = 3
NORMALIZE_MODE_2   = 'none'     # solo para visualizar
SPECTRUM_ALPHA_2   = 1
X_MIN_2 = 700.0
X_MAX_2 = 4500.0
IMG_NAME_2 = os.path.join(IMG_DIR_2, "spectrum_gauss_peaks_RECALC_annot.png")

# Resolución para decidir pares cercanos (medida sobre el propio espectro)
DEFAULT_RESOLUTION_EV = 5.0
USE_MEASURED_RES_FOR_PAIRING = True
RES_SEARCH_WINDOW_EV   = 40.0
RES_SNR_THRESHOLD      = 5.0
RES_MIN_ISOLATION_DROP = 0.50
RES_SUMMARY_STAT       = 'median'      # 'median'|'mean'|'best'

# =============================================================================
# ESTADO COMPARTIDO EN MEMORIA
# =============================================================================
# Mapa: clave = energía SyW (redondeada a 3 decimales), valor = %W en ese pico
PCT_MAP_MEMORY = {}

# =============================================================================
# UTILIDADES COMUNES
# =============================================================================
def make_uniform_edges_from_nbins(xmin, xmax, n_bins):
    xmin = float(xmin); xmax = float(xmax); n_bins = int(n_bins)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin or n_bins < 1:
        raise ValueError("Rango o NUM_BINS inválidos.")
    return np.linspace(xmin, xmax, n_bins + 1, dtype=float)

def centers_to_edges_clamped(centers, x_min, x_max):
    c = np.asarray(centers, dtype=float)
    n = c.size
    if n == 0:
        return np.array([x_min, x_max], dtype=float)
    edges = np.empty(n + 1, dtype=float)
    if n == 1:
        w = (x_max - x_min)
        edges[0] = c[0] - 0.5 * w
        edges[1] = c[0] + 0.5 * w
    else:
        mids = 0.5 * (c[:-1] + c[1:])
        edges[1:-1] = mids
        edges[0]  = c[0]  - (mids[0]  - c[0])
        edges[-1] = c[-1] + (c[-1]    - mids[-1])
    edges = np.clip(edges, x_min, x_max)
    edges = np.maximum.accumulate(edges)
    return edges

def rebin_counts_to_edges(src_edges, src_counts, tgt_edges):
    src_edges = np.asarray(src_edges, dtype=float)
    src_counts = np.asarray(src_counts, dtype=float)
    tgt_edges = np.asarray(tgt_edges, dtype=float)
    nS = len(src_counts); nT = len(tgt_edges) - 1
    out = np.zeros(nT, dtype=float); i = j = 0
    while i < nS and j < nT:
        Ls, Rs = src_edges[i], src_edges[i+1]
        Lt, Rt = tgt_edges[j], tgt_edges[j+1]
        if Rs <= Lt: i += 1; continue
        if Rt <= Ls: j += 1; continue
        overlap = min(Rs, Rt) - max(Ls, Lt)
        if overlap > 0:
            width_src = max(Rs - Ls, 1e-12)
            out[j] += src_counts[i] * (overlap / width_src)
        if Rs <= Rt: i += 1
        else:        j += 1
    return out

def read_peaks_list(path_txt):
    try:
        with open(path_txt, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]
    except Exception:
        return []

# -------------------- ajuste gauss + fondo --------------------
def _gauss(E, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5*((E - mu)/sigma)**2)

def _gauss_lin(E, p):
    b0, b1, A, mu, sigma = p
    return b0 + b1*E + A * _gauss(E, mu, sigma)

def _residuals(p, E, Y):
    return _gauss_lin(E, p) - Y

def _local_background(x,y,mu_guess,window):
    mask = (x > mu_guess - window) & (x < mu_guess + window)
    if not np.any(mask): return 0.0,0.0
    xv,yv = x[mask],y[mask]
    core = (xv > mu_guess - window/4) & (xv < mu_guess + window/4)
    xb,yb = xv[~core], yv[~core]
    if len(xb) < 3: xb,yb = xv,yv
    A = np.vstack([np.ones_like(xb), (xb - mu_guess)]).T
    coeff,_,_,_ = np.linalg.lstsq(A, yb, rcond=None)
    return float(coeff[0]), float(coeff[1])

def fit_gauss_window(Ew, Yw, mu0, sigma0=None, fwhm_hint=None):
    Ew = np.asarray(Ew, dtype=float); Yw = np.asarray(Yw, dtype=float)
    dEw = max(np.median(np.diff(Ew)), 1e-12)
    if sigma0 is None:
        sigma0 = max((Ew.max()-Ew.min())/8.0, 2.0*dEw)
    if fwhm_hint is not None and np.isfinite(fwhm_hint):
        sigma_hint = max(fwhm_hint/2.355, 2.0*dEw)
        sigma_min = 0.5*sigma_hint
        sigma_max = 2.0*sigma_hint
    else:
        sigma_min = 2.0*dEw
        sigma_max = (Ew.max()-Ew.min())/2.0
    B0, B1 = _local_background(Ew, Yw, mu0, window=(Ew.max()-Ew.min()))
    A_init = max(1e-12, float(np.max(Yw - (B0 + B1*(Ew - mu0)))))
    p0 = np.array([B0, B1, min(A_init, np.max(Yw) if np.max(Yw)>0 else A_init), mu0,
                   min(max(sigma0, sigma_min), sigma_max)], dtype=float)
    bounds = (np.array([-np.inf, -np.inf, 0.0, mu0-2.0, sigma_min]),
              np.array([ np.inf,  np.inf, np.inf, mu0+2.0, sigma_max]))
    res = least_squares(_residuals, p0, args=(Ew, Yw), loss='huber', f_scale=1.0,
                        bounds=bounds, max_nfev=20000)
    p = res.x
    return {
        'success': bool(res.success),
        'params': {'b0': p[0], 'b1': p[1], 'A': p[2], 'mu': p[3], 'sigma': p[4]},
        'model': _gauss_lin(Ew, p),
        'Ew': Ew
    }

# -------------------- áreas (cuentas) --------------------
def gaussian_area_counts(A, sigma, binw):
    if not (np.isfinite(A) and np.isfinite(sigma) and sigma > 0 and binw > 0):
        return np.nan
    return float((A * sigma * math.sqrt(2.0*math.pi)) / binw)

# -------------------- pairing --------------------
def pair_peaks_by_nearest(peaks_W, peaks_S, tol_ev):
    """Para cada pico de S devuelve el pico W más cercano si |ΔE|<=tol_ev."""
    W = np.asarray(peaks_W, float); W.sort()
    pairs = []
    for es in peaks_S:
        if W.size == 0: break
        j = int(np.argmin(np.abs(W - es)))
        if abs(W[j] - es) <= tol_ev:
            pairs.append((float(W[j]), float(es)))
    return pairs

# =============================================================================
# PRIMERA PARTE: RATIO de áreas (sin solape), % por pico EN MEMORIA y figura
# =============================================================================
def run_ratio_and_export():
    global PCT_MAP_MEMORY
    PCT_MAP_MEMORY = {}  # reset

    # --- Carga W (eventos) y SyW (E,counts) ---
    dataA = np.load(PATH_DATA_A_NPY)
    E_beam, E_photon = dataA[1,:].astype(float), dataA[2,:].astype(float)

    arrB = np.load(PATH_DATA_B_NPY)
    if arrB.ndim != 2:
        return  # silencioso: no imprimir nada salvo el global al final de todo
    if arrB.shape[1] == 2:
        E_B_all, C_B_all = arrB[:,0].astype(float), arrB[:,1].astype(float)
    else:
        E_B_all, C_B_all = arrB[0,:].astype(float), arrB[1,:].astype(float)

    # --- ORDENAR SyW por energía antes de rebinnear ---
    idxB = np.argsort(E_B_all)
    E_B_all = E_B_all[idxB]
    C_B_all = C_B_all[idxB]

    # --- Filtro por E_beam ---
    mask_beam = np.abs(E_beam - TARGET_BEAM_ENERGY_EV) <= BEAM_TOLERANCE_EV
    Eph_beam = E_photon[mask_beam]
    if Eph_beam.size == 0:
        return

    # --- Malla única ---
    edges = make_uniform_edges_from_nbins(X_MIN_EV, X_MAX_EV, NUM_BINS)
    centers = 0.5*(edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]

    # --- Histogramas crudos ---
    counts_A, _ = np.histogram(Eph_beam, bins=edges)  # W
    mask_B = (E_B_all >= X_MIN_EV) & (E_B_all <= X_MAX_EV)
    counts_B = rebin_counts_to_edges(centers_to_edges_clamped(E_B_all[mask_B], X_MIN_EV, X_MAX_EV),
                                     C_B_all[mask_B], edges)  # S

    # --- Normalización base (peak) SIEMPRE ---
    if NORMALIZE_MODE.lower() == "peak":
        L = PEAK_CENTER_EV - PEAK_WINDOW_EV; R = PEAK_CENTER_EV + PEAK_WINDOW_EV
        win_edges = np.array([L, R], dtype=float)
        areaA = float(np.sum(rebin_counts_to_edges(edges, counts_A, win_edges)))
        areaB = float(np.sum(rebin_counts_to_edges(edges, counts_B, win_edges)))
        sA_base = 1.0/areaA if areaA > 0 else 1.0
        sB_base = 1.0/areaB if areaB > 0 else 1.0
    else:
        sA_base = sB_base = 1.0
    counts_A = counts_A * sA_base
    counts_B = counts_B * sB_base

    # --- Normalización adicional SOLO para el plot ---
    def plot_scale(mode, y):
        if mode == "max": return 1.0/np.max(y) if np.max(y) > 0 else 1.0
        if mode in ("area","pdf","density"): return 1.0/np.sum(y) if np.sum(y) > 0 else 1.0
        return 1.0
    sA_plot = plot_scale(PLOT_NORMALIZE_MODE, counts_A)
    sB_plot = plot_scale(PLOT_NORMALIZE_MODE, counts_B)
    yA = counts_A * sA_plot
    yB = counts_B * sB_plot

    # --- Emparejado por proximidad y ventanas independientes ---
    peaks_W   = read_peaks_list(PATH_PEAKS_W_TXT)   # picos W
    peaks_SyW = read_peaks_list(PATH_PEAKS_SYW_TXT) # picos S
    if len(peaks_W) == 0 or len(peaks_SyW) == 0:
        return

    tol_pair = max(MATCH_MAX_DELTA_EV, PAIR_TOL_FACTOR * FWHM_HINT_EV)
    pairs = pair_peaks_by_nearest(peaks_W, peaks_SyW, tol_pair)
    if not pairs:
        return

    # --- Plot base ---
    plt.figure(figsize=(12,6))
    plt.plot(centers, yA, label=LEGEND_W.format(beam=TARGET_BEAM_ENERGY_EV, tol=int(BEAM_TOLERANCE_EV)),
             linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA, color=COLOR_W)
    plt.plot(centers, yB, label=LEGEND_SYW, linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA, color=COLOR_S)

    used_legend_fit_W = used_legend_fit_S = False
    used_legend_point_W = used_legend_point_S = False

    # --- % por pico EN MEMORIA (sin escribir a disco) ---
    for ew, es in pairs:
        # Ventanas independientes centradas en cada pico emparejado
        Emin_W = ew - WINDOW_MARGIN_EV; Emax_W = ew + WINDOW_MARGIN_EV
        Emin_S = es - WINDOW_MARGIN_EV; Emax_S = es + WINDOW_MARGIN_EV

        mask_W = (centers >= Emin_W) & (centers <= Emax_W)
        mask_S = (centers >= Emin_S) & (centers <= Emax_S)

        if np.count_nonzero(mask_W) < 5 or np.count_nonzero(mask_S) < 5:
            continue

        Ew_W = centers[mask_W]; YW = counts_A[mask_W].astype(float)  # W normalizado (peak) en ventana W
        Ew_S = centers[mask_S]; YS = counts_B[mask_S].astype(float)  # S normalizado (peak) en ventana S

        # Ajustes con mu inicial en su propio pico; pista de FWHM fija
        fitW = fit_gauss_window(Ew_W, YW, mu0=ew, fwhm_hint=FWHM_HINT_EV)
        fitS = fit_gauss_window(Ew_S, YS, mu0=es, fwhm_hint=FWHM_HINT_EV)

        pW = fitW['params']; pS = fitS['params']
        AW, muW, sigW = float(pW['A']), float(pW['mu']), float(pW['sigma'])
        AS, muS, sigS = float(pS['A']), float(pS['mu']), float(pS['sigma'])

        # Áreas (cuentas) TRAS normalización 'peak'
        area_W = gaussian_area_counts(AW, sigW, binw)
        area_S = gaussian_area_counts(AS, sigS, binw)

        # Ratio de áreas -> %W (cap 100)
        if np.isfinite(area_W) and np.isfinite(area_S) and area_S > 0:
            pct_W_in_S = 100.0 * float(np.clip(area_W / area_S, 0.0, 1.0))
            # Guarda en memoria (clave = energía SyW del TXT, redondeada a 3 decimales)
            PCT_MAP_MEMORY[round(float(es), 3)] = pct_W_in_S

        # Curvas para dibujar (cada una en su propio eje)
        b0W, b1W = pW['b0'], pW['b1']; b0S, b1S = pS['b0'], pS['b1']
        Gw_plot = (AW * _gauss(Ew_W, muW, sigW)) * sA_plot
        Gs_plot = (AS * _gauss(Ew_S, muS, sigS)) * sB_plot
        YgW = (b0W + b1W*Ew_W) * sA_plot + Gw_plot
        YgS = (b0S + b1S*Ew_S) * sB_plot + Gs_plot

        plt.plot(Ew_W, YgW, '--', linewidth=2.0, color=COLOR_FIT_W,
                 label=(LEGEND_FIT_W if not used_legend_fit_W else None), alpha=0.95)
        plt.plot(Ew_S, YgS, '--', linewidth=2.0, color=COLOR_FIT_S,
                 label=(LEGEND_FIT_SYW if not used_legend_fit_S else None), alpha=0.95)
        used_legend_fit_W = True
        used_legend_fit_S = True

        # Máximos de cada gaussiana (W y S)
        jW = int(np.argmax(Gw_plot)); xW = float(Ew_W[jW]); yW = float((b0W + b1W*xW)*sA_plot + Gw_plot[jW])
        jS = int(np.argmax(Gs_plot)); xS = float(Ew_S[jS]); yS = float((b0S + b1S*xS)*sB_plot + Gs_plot[jS])
        plt.scatter([xW],[yW], label=(LEGEND_MU_W if not used_legend_point_W else None), **MARKER_W)
        plt.scatter([xS],[yS], label=(LEGEND_MU_SYW if not used_legend_point_S else None), **MARKER_S)
        used_legend_point_W = True
        used_legend_point_S = True

        # Etiqueta junto al pico de S (energía y % si procede)
        if np.isfinite(muS):
            txt = f"{muS:.1f} eV"
            if round(float(es),3) in PCT_MAP_MEMORY:
                pct_val = PCT_MAP_MEMORY[round(float(es),3)]
                pct_str = f"<{DISPLAY_MIN_PERCENT:.2f}% W" if (0.0 < pct_val < DISPLAY_MIN_PERCENT) else f"{pct_val:.2f}% W"
                txt += f"\n{pct_str}"
            plt.text(xS, yS, txt, fontsize=9, color="black", ha='center', va='bottom')

    # --- cierre plot ---
    plt.xlim(X_MIN_EV, X_MAX_EV)
    plt.xlabel(XLABEL_PART1)
    plt.ylabel(YLABEL_PART1 if PLOT_NORMALIZE_MODE=='none' else "Counts (scaled)")
    plt.title(TITLE_PART1)
    plt.legend(); plt.tight_layout()
    fplot = os.path.join(OUTPUT_DIR, f"W_vs_SyW_RATIO_{NUM_BINS}bins_{int(X_MIN_EV)}-{int(X_MAX_EV)}.png")
    plt.savefig(fplot, dpi=DPI_SAVE)
    plt.show()   # ← mostrar en pantalla

# =============================================================================
# SEGUNDA PARTE: espectro único + gaussianas (usa %W del mapa en memoria)
# =============================================================================
def normalization_factor(counts, mode, bin_width=None):
    if counts.size == 0: return 1.0
    mode = (mode or 'none').lower()
    if mode == 'none': return 1.0
    if mode in ('area','pdf','density'):
        total = float(np.sum(counts))
        if bin_width is not None: total *= float(bin_width)
        return (1.0/total) if total>0 else 1.0
    if mode == 'max':
        m = float(np.max(counts)); return (1.0/m) if m>0 else 1.0
    return 1.0

def gauss_lin(E, p):
    b0,b1,A,mu,sigma = p
    sigma = max(float(sigma), 1e-12)
    return b0 + b1*E + A*np.exp(-0.5*((E-mu)/sigma)**2)

def residuals_gauss_lin(p, E, Y):
    return gauss_lin(E,p)-Y

def fit_single_gaussian_window(Ew, Yw, mu0=None, sigma0=None):
    if mu0 is None:
        w = np.clip(Yw - np.percentile(Yw, 20), 0, None); sw = np.sum(w)
        mu0 = np.sum(w*Ew)/sw if sw>0 else Ew[np.argmax(Yw)]
    if sigma0 is None:
        sigma0 = max((Ew.max()-Ew.min())/8.0, max(np.median(np.diff(Ew)), 1e-12)*2)
    A_ls = np.vstack([Ew, np.ones_like(Ew)]).T
    b1_init, b0_init = np.linalg.lstsq(A_ls, Yw, rcond=None)[0]
    # amplitud robusta con fondo local
    B0, B1 = _local_background(Ew, Yw, mu0, window=(Ew.max()-Ew.min()))
    A_init = max(1e-12, float(np.max(Yw - (B0 + B1*(Ew - mu0)))))
    p0 = np.array([b0_init, b1_init, A_init, mu0, sigma0], dtype=float)
    dEw = max(np.median(np.diff(Ew)), 1e-12)
    bounds = (np.array([-np.inf, -np.inf, 0.0, Ew.min(), 2.0*dEw]),
              np.array([ np.inf,  np.inf, np.inf, Ew.max(), (Ew.max()-Ew.min())*0.5]))
    res = least_squares(residuals_gauss_lin, p0, args=(Ew, Yw),
                        loss='soft_l1', f_scale=1.0, bounds=bounds, max_nfev=20000)
    p = res.x
    return {'success':res.success,'params':{'b0':p[0],'b1':p[1],'A':p[2],'mu':p[3],'sigma':p[4]},
            'model_curve':gauss_lin(Ew,p),'Ew':Ew}

def _half_max_width(x,y):
    if len(x)<4 or np.all(y<=0): return np.nan,np.nan,False
    imax=int(np.argmax(y)); ymax=float(y[imax]); half=ymax/2.0
    i=imax
    while i>0 and y[i]>half: i-=1
    if i==0: return np.nan,np.nan,False
    xL = x[i] + (x[i+1]-x[i])*(half - y[i])/(y[i+1]-y[i] + 1e-12)
    j=imax
    while j<len(y)-1 and y[j]>half: j+=1
    if j>=len(y)-1: return np.nan,np.nan,False
    xR = x[j-1] + (x[j]-x[j-1])*(half - y[j-1])/(y[j]-y[j-1] + 1e-12)
    return float(x[imax]), float(xR-xL), True

def measure_resolution_from_peaklist(E, C, peak_list):
    results=[]
    if E.size<5 or C.size<5 or np.max(C)<=0: return results,(np.nan,0)
    med=np.median(C); mad=np.median(np.abs(C-med))+1e-9; sigma_bg=1.4826*mad
    for e0 in peak_list:
        mask=(E>e0-RES_SEARCH_WINDOW_EV)&(E<e0+RES_SEARCH_WINDOW_EV)
        x=E[mask]; y=C[mask]
        if x.size<8: continue
        B0,B1=_local_background(E,C,e0,window=RES_SEARCH_WINDOW_EV)
        ycorr=y-(B0+B1*(x-e0)); ycorr[ycorr<0]=0.0
        core=(x>e0-RES_SEARCH_WINDOW_EV/6)&(x<e0+RES_SEARCH_WINDOW_EV/6)
        resid=y[~core]-(B0+B1*(x[~core]-e0))
        sigma_loc=np.std(resid) if resid.size>=5 else sigma_bg
        height=float(np.max(ycorr)); SNR=height/(sigma_loc+1e-9)
        if SNR<RES_SNR_THRESHOLD: continue
        imax=int(np.argmax(ycorr))
        left=min(ycorr[:imax+1]) if imax>0 else height
        right=min(ycorr[imax:]) if imax<len(ycorr)-1 else height
        valley=max(left,right)
        if not (valley <= RES_MIN_ISOLATION_DROP*height): continue
        mu,fwhm,ok=_half_max_width(x,ycorr)
        if not ok or not np.isfinite(fwhm) or fwhm<=0: continue
    results.append({'E_peak':mu,'FWHM_eV':fwhm,'rel_resolution':fwhm/mu,'SNR':SNR,'ok':True})
    if not results: return results,(np.nan,0)
    vals=np.array([r['FWHM_eV'] for r in results],float)
    if RES_SUMMARY_STAT=='mean': summary=(float(np.mean(vals)),len(vals))
    elif RES_SUMMARY_STAT=='best': summary=(float(np.min(vals)),len(vals))
    else: summary=(float(np.median(vals)),len(vals))
    results.sort(key=lambda r:r['E_peak'])
    return results,summary

def estimate_peak_counts_from_fit(A, sigma, bin_w, Ew_fit=None, mu=None):
    if bin_w is not None and bin_w>0 and np.isfinite(A) and np.isfinite(sigma):
        return float((A * sigma * math.sqrt(2.0*math.pi)) / bin_w)
    if Ew_fit is not None and mu is not None and np.isfinite(A) and np.isfinite(sigma):
        g = A * np.exp(-0.5*((Ew_fit - mu)/max(sigma, 1e-12))**2)
        return float(np.sum(g))
    return float('nan')

def run_single_spectrum_with_annotations():
    # Lectura del espectro
    arr = np.load(PATH_SPECTRUM_NPY)
    if arr.ndim != 2 or (arr.shape[1] != 2 and arr.shape[0] != 2):
        return  # sin prints
    if arr.shape[1]==2:
        E_all = np.asarray(arr[:,0],float); C_all = np.asarray(arr[:,1],float)
    else:
        E_all = np.asarray(arr[0,:],float); C_all = np.asarray(arr[1,:],float)
    idx = np.argsort(E_all); E_all=E_all[idx]; C_all=C_all[idx]
    mask = (E_all>=X_MIN_2) & (E_all<=X_MAX_2)
    E_centers = E_all[mask]; counts_raw = C_all[mask]
    if E_centers.size==0: return
    bin_width = float(np.median(np.diff(E_centers))) if E_centers.size>=2 else None

    # Lista de picos desde TXT (entrada)
    try:
        with open(PATH_PEAKS_TXT,'r') as f:
            peaks_txt = [float(line.strip()) for line in f if line.strip()]
    except Exception:
        peaks_txt = []
    peaks = sorted([e for e in peaks_txt if (e>=E_centers.min() and e<=E_centers.max())])

    # Medida de resolución (solo para agrupar pares; no imprime nada)
    res_list, (fwhm_summary, n_used) = measure_resolution_from_peaklist(E_centers, counts_raw, peaks)
    RES_USED_FOR_PAIRING = float(fwhm_summary) if (USE_MEASURED_RES_FOR_PAIRING and np.isfinite(fwhm_summary)) else float(DEFAULT_RESOLUTION_EV)

    # Normalización para el plot (no altera el ajuste)
    s_norm = normalization_factor(counts_raw, NORMALIZE_MODE_2, bin_width=bin_width)
    counts_plot = counts_raw * s_norm

    # Ajustes gaussianos
    gaussians_to_plot = []
    final_E, final_Y_norm = [], []
    peak_count_rows = []  # (E_txt, mu_fit, A, sigma, cuentas_estimadas, x_peak, y_peak)

    i=0
    N=len(peaks)
    while i<N:
        e0 = peaks[i]
        # Pares cercanos
        if i+1<N and abs(peaks[i+1]-e0) < RES_USED_FOR_PAIRING:
            e1 = peaks[i+1]
            Emin=min(e0,e1)-WINDOW_MARGIN_EV_2; Emax=max(e0,e1)+WINDOW_MARGIN_EV_2
            mask_win=(E_centers>=Emin)&(E_centers<=Emax)
            Ew=E_centers[mask_win]; Yw=counts_raw[mask_win]
            if Ew.size>=5 and np.any(Yw>0):
                w=np.clip(Yw - np.percentile(Yw,20),0,None); sw=np.sum(w)
                mu0=(np.sum(w*Ew)/sw) if sw>0 else 0.5*(e0+e1)
                sigma0=max((Emax-Emin)/8.0, max(np.median(np.diff(Ew)),1e-12)*2)
                fit=fit_single_gaussian_window(Ew,Yw,mu0=mu0, sigma0=sigma0)
                pars=fit['params']; A=float(pars['A']); mu_fit=float(pars['mu']); sigma_fit=float(pars['sigma'])
                peak_counts = estimate_peak_counts_from_fit(A, sigma_fit, bin_width, Ew_fit=fit['Ew'], mu=mu_fit)

                Yg_norm=fit['model_curve']*s_norm; Ew_fit=fit['Ew']; jmax=int(np.argmax(Yg_norm))
                x_peak=float(Ew_fit[jmax]); y_peak=float(Yg_norm[jmax])

                gaussians_to_plot.append((Ew_fit, Yg_norm))
                final_E.append(x_peak); final_Y_norm.append(y_peak)

                peak_count_rows.append((e0, mu_fit, A, sigma_fit, peak_counts, x_peak, y_peak))
                peak_count_rows.append((e1, mu_fit, A, sigma_fit, peak_counts, x_peak, y_peak))
            i+=2
        else:
            # Pico aislado
            Emin=e0-WINDOW_MARGIN_EV_2; Emax=e0+WINDOW_MARGIN_EV_2
            mask_win=(E_centers>=Emin)&(E_centers<=Emax)
            Ew=E_centers[mask_win]; Yw=counts_raw[mask_win]
            if Ew.size>=5 and np.any(Yw>0):
                fit=fit_single_gaussian_window(Ew,Yw,mu0=e0,
sigma0=max((Emax-Emin)/8.0, max(np.median(np.diff(Ew)),1e-12)*2))
                pars=fit['params']; A=float(pars['A']); mu_fit=float(pars['mu']); sigma_fit=float(pars['sigma'])
                peak_counts = estimate_peak_counts_from_fit(A, sigma_fit, bin_width, Ew_fit=fit['Ew'], mu=mu_fit)

                Yg_norm=fit['model_curve']*s_norm; Ew_fit=fit['Ew']; jmax=int(np.argmax(Yg_norm))
                x_peak=float(Ew_fit[jmax]); y_peak=float(Yg_norm[jmax])

                gaussians_to_plot.append((Ew_fit, Yg_norm))
                final_E.append(x_peak); final_Y_norm.append(y_peak)

                peak_count_rows.append((e0, mu_fit, A, sigma_fit, peak_counts, x_peak, y_peak))
            i+=1

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(E_centers, counts_plot, label=LEGEND_SPEC, color='#1f77b4',
             linewidth=1.2, alpha=SPECTRUM_ALPHA_2, zorder=3)

    gauss_label_used=False
    for Ew_g, Yg_norm in gaussians_to_plot:
        if not gauss_label_used:
            plt.plot(Ew_g, Yg_norm, '--', color='#39FF14', alpha=SPECTRUM_ALPHA_2,
                     linewidth=2.0, zorder=10, label=LEGEND_FIT)
            gauss_label_used=True
        else:
            plt.plot(Ew_g, Yg_norm, '--', color='#39FF14', alpha=SPECTRUM_ALPHA_2,
                     linewidth=2.0, zorder=10)

    # Máximos y etiquetas de energía (sin %)
    if final_E:
        plt.scatter(final_E, final_Y_norm, color='red', s=16, zorder=12,
                    edgecolor='white', linewidth=0.5, label=LEGEND_MAX)
        for E_txt, mu_fit, A, sigma, c_est, x_peak, y_peak in peak_count_rows:
            plt.text(x_peak, y_peak, f"{x_peak:.1f} eV", fontsize=8, color='black',
                     ha='center', va='bottom', zorder=15)

    plt.xlabel(XLABEL_PART2)
    plt.ylabel(YLABEL_PART2 if NORMALIZE_MODE_2.lower()=='none' else ("Probability density (1/eV)" if NORMALIZE_MODE_2.lower() in ('area','pdf','density') else "Normalized counts (max=1)"))
    plt.title(TITLE_PART2)
    plt.xlim(X_MIN_2, X_MAX_2); plt.legend(); plt.tight_layout()
    plt.savefig(IMG_NAME_2, dpi=200)
    plt.show()   # ← mostrar en pantalla

    # ===== SOLO MENSAJE FINAL: porcentaje global de W =====
    if peak_count_rows:
        tungsten_counts_total = 0.0
        for E_txt, mu_fit, A, sigma, c_est, x_peak, y_peak in peak_count_rows:
            # %W por pico desde memoria; si no está, 100%
            pct = PCT_MAP_MEMORY.get(round(float(E_txt), 3), 100.0)
            factor = (pct/100.0) if np.isfinite(pct) else 1.0
            c_base = c_est if (c_est is not None and np.isfinite(c_est)) else 0.0
            tungsten_counts_total += factor * c_base

        total_counts = float(np.sum(counts_raw))
        pct_global = 100.0 * tungsten_counts_total / total_counts if total_counts>0 else float('nan')

        # ÚNICA LÍNEA QUE SE IMPRIME:
        print(f"{pct_global:.3f}")
    else:
        # Si no hay picos válidos, imprimimos 0.000 para no dejar vacío
        print("0.000")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    run_ratio_and_export()                 # 1ª figura + % por pico en memoria (sin archivos)
    run_single_spectrum_with_annotations() # 2ª figura y SOLO imprime el % global (una línea)
