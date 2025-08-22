import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.optimize import least_squares
import os
import math
import csv

# ===================== RUTAS / ENTRADAS COMUNES =====================
# --- PRIMERA PARTE (SyW vs W por solape; deja la gráfica tal cual, y exporta % por pico) ---
PATH_DATA_A_NPY    = "/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy" # W puro
PATH_DATA_B_NPY    = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run002_40pix.npy" # SyW
PATH_PEAKS_W_TXT   = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/W.txt"
PATH_PEAKS_SYW_TXT = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/SyW.txt"

# --- SEGUNDA PARTE (un solo espectro + gaussianas) ---
PATH_SPECTRUM_NPY  = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run002_40pix.npy"
PATH_PEAKS_TXT     = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones/COMPARAR.txt"
IMG_DIR_2          = "/home/lopezgonza/Desktop/FEBIT/Figuras/Prueba/UnEspectro"

# --- Salidas ---
OUTPUT_DIR         = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR_2, exist_ok=True)
CSV_PCT_BY_PEAK    = os.path.join(OUTPUT_DIR, "pct_W_por_pico_SyW.csv")   # puente entre 1ª y 2ª parte

# ===================== PARÁMETROS 1ª PARTE (SyW vs W) =====================
TARGET_BEAM_ENERGY_EV = 3800.0
BEAM_TOLERANCE_EV     = 50.0
X_MIN_EV   = 700.0
X_MAX_EV   = 4500.0
NUM_BINS   = 3000
NORMALIZE_MODE = "peak"   # "peak" | "area" | "max" | "none"
PEAK_CENTER_EV  = 2179.8
PEAK_WINDOW_EV  = 6.0
PLOT_NORMALIZE_MODE = "none"  # solo para dibujar
MATCH_MAX_DELTA_EV = 6.0
WINDOW_MARGIN_EV   = 4.0
DISPLAY_MIN_PERCENT = 0.01
ASSUME_100_IF_NO_MATCH = True
ASSUME_100_IF_NAN      = True
SPECTRUM_ALPHA = 0.85
LINEWIDTH = 1.6
DPI_SAVE = 200
COLOR_W = '#1f77b4'
COLOR_S = '#ff7f0e'
COLOR_FIT_W = '#2ca02c'
COLOR_FIT_S = '#d62728'
MARKER_W = dict(marker='o', s=20, color=COLOR_FIT_W, edgecolor='white', linewidth=0.6, zorder=12)
MARKER_S = dict(marker='o', s=22, color=COLOR_FIT_S, edgecolor='white', linewidth=0.6, zorder=12)

# ===================== PARÁMETROS 2ª PARTE (un espectro) =====================
WINDOW_MARGIN_EV_2 = 5
NORMALIZE_MODE_2   = 'none'            # 'area' | 'max' | 'none' (solo para ploteo)
SPECTRUM_ALPHA_2   = 1
ANNOTATE_PEAK_LABELS = True
DEFAULT_RESOLUTION_EV = 5.0
USE_MEASURED_RES_FOR_PAIRING = True
RES_SEARCH_WINDOW_EV   = 40.0
RES_SNR_THRESHOLD      = 5.0
RES_MIN_ISOLATION_DROP = 0.50
RES_SUMMARY_STAT       = 'median'      # 'median'|'mean'|'best'
X_MIN_2 = 700.0
X_MAX_2 = 4500.0
IMG_NAME_2 = os.path.join(IMG_DIR_2, "spectrum_gauss_peaks_RECALC_annot.png")

# ===================== UTILIDADES COMUNES =====================
def make_uniform_edges_from_nbins(xmin, xmax, n_bins):
    xmin = float(xmin); xmax = float(xmax); n_bins = int(n_bins)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin or n_bins < 1:
        raise ValueError("Rango o NUM_BINS inválidos.")
    return np.linspace(xmin, xmax, n_bins + 1, dtype=float)

def centers_to_edges_clamped(centers, x_min, x_max):
    c = np.asarray(centers, dtype=float)
    n = c.size
    if n == 0: return np.array([x_min, x_max], dtype=float)
    edges = np.empty(n + 1, dtype=float)
    if n == 1:
        w = (x_max - x_min)
        edges[0] = c[0] - 0.5 * w; edges[1] = c[0] + 0.5 * w
    else:
        mids = 0.5 * (c[:-1] + c[1:]); edges[1:-1] = mids
        edges[0]  = c[0]  - (mids[0]  - c[0]); edges[-1] = c[-1] + (c[-1] - mids[-1])
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

def split_B_energy_counts(v1, v2):
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    def score_as_energy(x):
        x = np.asarray(x, dtype=float); uniq = np.unique(x)
        return (uniq.size, np.ptp(uniq))
    s1 = score_as_energy(v1); s2 = score_as_energy(v2)
    pick_v1_as_E = (s1[0] > s2[0]) or (s1[0] == s2[0] and s1[1] >= s2[1])
    return (v1, v2, "E=v1, C=v2") if pick_v1_as_E else (v2, v1, "E=v2, C=v1")

def _gauss_lin(E, p):
    b0, b1, A, mu, sigma = p
    sigma = max(float(sigma), 1e-12)
    return b0 + b1*E + A * np.exp(-0.5*((E - mu)/sigma)**2)

def _residuals(p, E, Y):
    return _gauss_lin(E, p) - Y

def fit_gauss_window(Ew, Yw, mu0, sigma0=None):
    Ew = np.asarray(Ew, dtype=float); Yw = np.asarray(Yw, dtype=float)
    if sigma0 is None:
        sigma0 = max((Ew.max()-Ew.min())/8.0, max(np.median(np.diff(Ew)), 1e-12)*2)
    A_ls = np.vstack([Ew, np.ones_like(Ew)]).T
    b1_init, b0_init = np.linalg.lstsq(A_ls, Yw, rcond=None)[0]
    A_init = max(Yw.max() - (b0_init + b1_init*mu0), (Yw.max()-np.median(Yw))*0.5)
    p0 = np.array([b0_init, b1_init, A_init, mu0, sigma0], dtype=float)
    dEw = max(np.median(np.diff(Ew)), 1e-12)
    bounds = (np.array([-np.inf, -np.inf, 0.0, Ew.min(), 0.5*dEw]),
              np.array([ np.inf,  np.inf, np.inf, Ew.max(), (Ew.max()-Ew.min())*2.0]))
    res = least_squares(_residuals, p0, args=(Ew, Yw), loss='soft_l1', f_scale=1.0, bounds=bounds, max_nfev=20000)
    p = res.x
    return {
        'success': bool(res.success),
        'params': {'b0': p[0], 'b1': p[1], 'A': p[2], 'mu': p[3], 'sigma': p[4]},
        'model': _gauss_lin(Ew, p),
        'Ew': Ew
    }

def gaussian_only(E, A, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return A * np.exp(-0.5*((E - mu)/sigma)**2)

def gaussian_area_counts(A, sigma, binw):
    if not (np.isfinite(A) and np.isfinite(sigma) and sigma > 0 and binw > 0):
        return np.nan
    return float((A * sigma * math.sqrt(2.0*math.pi)) / binw)

def overlap_area_counts(A1, mu1, s1, A2, mu2, s2, binw, L, R):
    if not all(np.isfinite(x) for x in [A1, mu1, s1, A2, mu2, s2, binw]):
        return np.nan
    if s1 <= 0 or s2 <= 0 or binw <= 0 or L >= R:
        return np.nan
    sig_min = min(s1, s2)
    dx = max(min(binw, sig_min/10.0), (R-L)/2000.0)
    x = np.arange(L, R + dx, dx)
    g1 = gaussian_only(x, A1, mu1, s1)
    g2 = gaussian_only(x, A2, mu2, s2)
    y = np.minimum(g1, g2)
    integral = np.trapz(y, x)
    return float(integral / binw)

def read_peaks_list(path_txt):
    try:
        with open(path_txt, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"[WARN] No se pudo leer {path_txt}: {e}")
        return []

def pair_peaks(W_peaks, S_peaks, max_delta):
    W_peaks = sorted(W_peaks)
    S_peaks = sorted(S_peaks)
    pairs = []
    used_W = set()
    for es in S_peaks:
        if len(W_peaks) == 0:
            pairs.append((es, None, None))
            continue
        dists = [abs(w - es) if i not in used_W else np.inf for i, w in enumerate(W_peaks)]
        j = int(np.argmin(dists))
        if dists[j] <= max_delta:
            pairs.append((es, W_peaks[j], dists[j]))
            used_W.add(j)
        else:
            pairs.append((es, None, None))
    return pairs

# ===================== PARTE 1: SyW vs W, cálculo % por pico + gráfica =====================
def run_overlap_and_export():
    dataA = np.load(PATH_DATA_A_NPY)
    E_beam, E_photon = dataA[1,:].astype(float), dataA[2,:].astype(float)
    arrB = np.load(PATH_DATA_B_NPY)
    if arrB.ndim != 2: raise ValueError("SyW debe ser 2D")
    if arrB.shape[1] == 2:
        E_B_all, C_B_all = arrB[:,0].astype(float), arrB[:,1].astype(float)
    else:
        E_B_all, C_B_all = arrB[0,:].astype(float), arrB[1,:].astype(float)

    mask_beam = np.abs(E_beam - TARGET_BEAM_ENERGY_EV) <= BEAM_TOLERANCE_EV
    Eph_beam = E_photon[mask_beam]
    if Eph_beam.size == 0:
        print("[INFO] No hay W en el rango de E_beam."); return

    edges = make_uniform_edges_from_nbins(X_MIN_EV, X_MAX_EV, NUM_BINS)
    centers = 0.5*(edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]

    counts_A, _ = np.histogram(Eph_beam, bins=edges)
    mask_B = (E_B_all >= X_MIN_EV) & (E_B_all <= X_MAX_EV)
    counts_B = rebin_counts_to_edges(centers_to_edges_clamped(E_B_all[mask_B], X_MIN_EV, X_MAX_EV),
                                     C_B_all[mask_B], edges)

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

    def plot_scale(mode, y):
        if mode=="max": return 1.0/np.max(y) if np.max(y)>0 else 1.0
        if mode in ("area","pdf","density"): return 1.0/np.sum(y) if np.sum(y)>0 else 1.0
        return 1.0
    sA_plot = plot_scale(PLOT_NORMALIZE_MODE, counts_A)
    sB_plot = plot_scale(PLOT_NORMALIZE_MODE, counts_B)
    yA = counts_A * sA_plot; yB = counts_B * sB_plot

    peaks_W   = read_peaks_list(PATH_PEAKS_W_TXT)
    peaks_SyW = read_peaks_list(PATH_PEAKS_SYW_TXT)
    pairs = pair_peaks(peaks_W, peaks_SyW, MATCH_MAX_DELTA_EV)

    plt.figure(figsize=(12,6))
    plt.plot(centers, yA, label=f"W @ {TARGET_BEAM_ENERGY_EV:.0f}eV (±{int(BEAM_TOLERANCE_EV)})",
             linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA, color=COLOR_W)
    plt.plot(centers, yB, label="SyW (rebinned)", linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA, color=COLOR_S)

    used_legend_fit_W = used_legend_fit_S = False
    used_legend_point_W = used_legend_point_S = False

    with open(CSV_PCT_BY_PEAK, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["E_SyW_txt_eV","mu_S_fit_eV","pct_W_en_S"])

        for es, ew, d in pairs:
            if not (X_MIN_EV <= es <= X_MAX_EV): continue
            Emin = es - WINDOW_MARGIN_EV; Emax = es + WINDOW_MARGIN_EV
            mask = (centers >= Emin) & (centers <= Emax)
            if np.count_nonzero(mask) < 5: continue

            Ew = centers[mask]
            YW = counts_A[mask].astype(float)
            YS = counts_B[mask].astype(float)

            mu0_W = ew if ew is not None else es
            fitW = fit_gauss_window(Ew, YW, mu0=mu0_W)
            fitS = fit_gauss_window(Ew, YS, mu0=es)

            pW = fitW['params']; pS = fitS['params']
            AW, muW, sigW = float(pW['A']), float(pW['mu']), float(pW['sigma'])
            AS, muS, sigS = float(pS['A']), float(pS['mu']), float(pS['sigma'])

            area_S = gaussian_area_counts(AS, sigS, binw)
            shared = overlap_area_counts(AW, muW, sigW, AS, muS, sigS, binw, Emin, Emax)
            if not np.isfinite(shared) or shared < 0: shared = 0.0
            pct_W_in_S = (100.0 * max(0.0, min(shared, area_S)) / area_S) if (np.isfinite(area_S) and area_S>0) else np.nan

            if (ASSUME_100_IF_NO_MATCH and ew is None) or (ASSUME_100_IF_NAN and not np.isfinite(pct_W_in_S)):
                pct_W_in_S = 100.0

            writer.writerow([f"{es:.6f}", f"{muS:.6f}", f"{pct_W_in_S:.6f}"])

            Gw_plot = gaussian_only(fitW['Ew'], AW, muW, sigW) * sA_plot
            Gs_plot = gaussian_only(fitS['Ew'], AS, muS, sigS) * sB_plot
            b0W, b1W = pW['b0'], pW['b1']; b0S, b1S = pS['b0'], pS['b1']
            YgW = (b0W + b1W*fitW['Ew']) * sA_plot + Gw_plot
            YgS = (b0S + b1S*fitS['Ew']) * sB_plot + Gs_plot

            plt.plot(fitW['Ew'], YgW, '--', linewidth=2.0, color=COLOR_FIT_W,
                     label=("Ajuste W" if not used_legend_fit_W else None), alpha=0.95)
            plt.plot(fitS['Ew'], YgS, '--', linewidth=2.0, color=COLOR_FIT_S,
                     label=("Ajuste SyW" if not used_legend_fit_S else None), alpha=0.95)
            used_legend_fit_W = used_legend_fit_S = True

            jW = int(np.argmax(Gw_plot)); xW = float(fitW['Ew'][jW]); yW = float((b0W + b1W*xW)*sA_plot + Gw_plot[jW])
            jS = int(np.argmax(Gs_plot)); xS = float(fitS['Ew'][jS]); yS = float((b0S + b1S*xS)*sB_plot + Gs_plot[jS])
            plt.scatter([xW],[yW], label=("μ W (fit)" if not used_legend_point_W else None), **MARKER_W)
            plt.scatter([xS],[yS], label=("μ SyW (fit)" if not used_legend_point_S else None), **MARKER_S)
            used_legend_point_W = used_legend_point_S = True

            txt = f"{muS:.1f} eV"
            if np.isfinite(pct_W_in_S):
                pct_str = f"<{DISPLAY_MIN_PERCENT:.2f}% W" if (0.0 < pct_W_in_S < DISPLAY_MIN_PERCENT) else f"{pct_W_in_S:.2f}% W"
                txt += f"\n{pct_str}"
            plt.text(xS, yS, txt, fontsize=9, color="black", ha='center', va='bottom')

            link = f" | match W={ew:.2f} eV (Δ={d:.2f} eV)" if ew is not None else " | sin match W"
            print(f"[PICO SyW @ {es:.3f} eV]{link}: mu_W={muW:.3f}, mu_S={muS:.3f}, %W={pct_W_in_S:.3f}")

    plt.xlim(X_MIN_EV, X_MAX_EV)
    plt.xlabel("Energía (eV)")
    plt.ylabel("Cuentas (tras normalización 'peak')" if PLOT_NORMALIZE_MODE=='none' else "Counts (scaled)")
    plt.title("W puro vs SyW — %W por área compartida (sin sombreado)")
    plt.legend(); plt.tight_layout()
    fplot = os.path.join(OUTPUT_DIR, f"W_vs_SyW_NOFILL_{NUM_BINS}bins_{int(X_MIN_EV)}-{int(X_MAX_EV)}.png")
    plt.savefig(fplot, dpi=DPI_SAVE); print(f"[OK] Guardado: {fplot}")
    plt.show()
    print(f"[OK] CSV exportado: {CSV_PCT_BY_PEAK}")

# ===================== PARTE 2: Un espectro + gaussianas + aplicar % SOLO a picos presentes en CSV =====================
def normalization_factor(counts, mode, bin_width=None):
    if counts.size == 0: return 1.0
    mode = (mode or 'none').lower()
    if mode == 'none': return 1.0
    if mode in ('area','pdf','density'):
        total = float(np.sum(counts));
        if bin_width is not None: total *= float(bin_width)
        return (1.0/total) if total>0 else 1.0
    if mode == 'max':
        m = float(np.max(counts)); return (1.0/m) if m>0 else 1.0
    return 1.0

def gauss_lin(E, p):
    b0,b1,A,mu,sigma = p
    sigma = max(float(sigma), 1e-12)
    return b0 + b1*E + A*np.exp(-0.5*((E-mu)/sigma)**2)

def residuals_gauss_lin(p, E, Y): return gauss_lin(E,p)-Y

def fit_single_gaussian_window(Ew, Yw, mu0=None, sigma0=None):
    if mu0 is None:
        w = np.clip(Yw - np.percentile(Yw, 20), 0, None); sw = np.sum(w)
        mu0 = np.sum(w*Ew)/sw if sw>0 else Ew[np.argmax(Yw)]
    if sigma0 is None:
        sigma0 = max((Ew.max()-Ew.min())/8.0, max(np.median(np.diff(Ew)), 1e-12)*2)
    A_ls = np.vstack([Ew, np.ones_like(Ew)]).T
    b1_init, b0_init = np.linalg.lstsq(A_ls, Yw, rcond=None)[0]
    A_init = max(Yw.max() - (b0_init + b1_init*mu0), (Yw.max()-np.median(Yw))*0.5)
    p0 = np.array([b0_init, b1_init, A_init, mu0, sigma0], dtype=float)
    dEw = max(np.median(np.diff(Ew)), 1e-12)
    bounds = (np.array([-np.inf, -np.inf, 0.0, Ew.min(), 0.5*dEw]),
              np.array([ np.inf,  np.inf, np.inf, Ew.max(), (Ew.max()-Ew.min())*2.0]))
    res = least_squares(residuals_gauss_lin, p0, args=(Ew, Yw),
                        loss='soft_l1', f_scale=1.0, bounds=bounds, max_nfev=20000)
    p = res.x
    return {'success':res.success,'params':{'b0':p[0],'b1':p[1],'A':p[2],'mu':p[3],'sigma':p[4]},
            'model_curve':gauss_lin(Ew,p),'Ew':Ew}

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
    arr = np.load(PATH_SPECTRUM_NPY)
    if arr.ndim != 2 or (arr.shape[1] != 2 and arr.shape[0] != 2):
        raise ValueError("El .npy debe tener dos columnas (E,counts) o dos filas.")
    if arr.shape[1]==2:
        E_all = np.asarray(arr[:,0],float); C_all = np.asarray(arr[:,1],float)
    else:
        E_all = np.asarray(arr[0,:],float); C_all = np.asarray(arr[1,:],float)
    idx = np.argsort(E_all); E_all=E_all[idx]; C_all=C_all[idx]
    mask = (E_all>=X_MIN_2) & (E_all<=X_MAX_2)
    E_centers = E_all[mask]; counts_raw = C_all[mask]
    if E_centers.size==0: raise ValueError("No hay datos en el rango.")
    bin_width = float(np.median(np.diff(E_centers))) if E_centers.size>=2 else None

    with open(PATH_PEAKS_TXT,'r') as f:
        peaks_txt = [float(line.strip()) for line in f if line.strip()]
    peaks = sorted([e for e in peaks_txt if (e>=E_centers.min() and e<=E_centers.max())])

    # Cargar %W por pico, aplicándolos SOLO a picos presentes en el CSV (emparejamiento estricto por E_txt redondeada)
    pct_map = {}  # clave: round(E_SyW_txt, 3) -> porcentaje
    if os.path.exists(CSV_PCT_BY_PEAK):
        with open(CSV_PCT_BY_PEAK,'r') as fcsv:
            rdr = csv.DictReader(fcsv)
            for row in rdr:
                try:
                    e_txt = float(row["E_SyW_txt_eV"])
                    pct   = float(row["pct_W_en_S"])
                    pct_map[round(e_txt, 3)] = pct
                except:
                    continue

    res_list, (fwhm_summary, n_used) = measure_resolution_from_peaklist(E_centers, counts_raw, peaks)
    RES_USED_FOR_PAIRING = float(fwhm_summary) if (USE_MEASURED_RES_FOR_PAIRING and np.isfinite(fwhm_summary)) else float(DEFAULT_RESOLUTION_EV)
    if np.isfinite(fwhm_summary):
        print(f"Resolución (FWHM) estimada: {fwhm_summary:.3f} eV (n={n_used})")
    else:
        print("Resolución (FWHM) estimada: no disponible")

    s_norm = normalization_factor(counts_raw, NORMALIZE_MODE_2, bin_width=bin_width)
    counts_plot = counts_raw * s_norm

    gaussians_to_plot=[]; final_E=[]; final_Y_norm=[]
    report_rows=[]; peak_count_rows=[]; gaussians_counts=[]
    total_counts_in_range = float(np.sum(counts_raw))

    # función que devuelve (pct, aplica_pct) según el CSV; si no está en CSV => 100 y no aplicar
    def get_pct_for_peak_strict(e_txt):
        key = round(float(e_txt), 3)
        if key in pct_map:
            return float(pct_map[key]), True   # aplicar solo si existe en CSV
        return 100.0, False                    # no aplicar (se deja al 100 %)

    i=0; N=len(peaks)
    while i<N:
        e0 = peaks[i]
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
                gaussians_counts.append({'mu':mu_fit,'A':A,'sigma':sigma_fit,'counts_est':peak_counts,'linked_txt':[e0,e1]})
                Yg_norm=fit['model_curve']*s_norm; Ew_fit=fit['Ew']; jmax=int(np.argmax(Yg_norm))
                x_peak=float(Ew_fit[jmax]); y_peak=float(Yg_norm[jmax])
                gaussians_to_plot.append((Ew_fit, Yg_norm))
                final_E.append(x_peak); final_Y_norm.append(y_peak)
                report_rows.append((e0, x_peak, x_peak-e0, y_peak))
                report_rows.append((e1, x_peak, x_peak-e1, y_peak))
                peak_count_rows.append((e0, mu_fit, A, sigma_fit, peak_counts))
                peak_count_rows.append((e1, mu_fit, A, sigma_fit, peak_counts))
            i+=2
        else:
            Emin=e0-WINDOW_MARGIN_EV_2; Emax=e0+WINDOW_MARGIN_EV_2
            mask_win=(E_centers>=Emin)&(E_centers<=Emax)
            Ew=E_centers[mask_win]; Yw=counts_raw[mask_win]
            if Ew.size>=5 and np.any(Yw>0):
                fit=fit_single_gaussian_window(Ew,Yw,mu0=e0,
sigma0=max((Emax-Emin)/8.0, max(np.median(np.diff(Ew)),1e-12)*2))
                pars=fit['params']; A=float(pars['A']); mu_fit=float(pars['mu']); sigma_fit=float(pars['sigma'])
                peak_counts = estimate_peak_counts_from_fit(A, sigma_fit, bin_width, Ew_fit=fit['Ew'], mu=mu_fit)
                gaussians_counts.append({'mu':mu_fit,'A':A,'sigma':sigma_fit,'counts_est':peak_counts,'linked_txt':[e0]})
                peak_count_rows.append((e0, mu_fit, A, sigma_fit, peak_counts))
                Yg_norm=fit['model_curve']*s_norm; Ew_fit=fit['Ew']; jmax=int(np.argmax(Yg_norm))
                x_peak=float(Ew_fit[jmax]); y_peak=float(Yg_norm[jmax])
                gaussians_to_plot.append((Ew_fit, Yg_norm))
                final_E.append(x_peak); final_Y_norm.append(y_peak)
                report_rows.append((e0, x_peak, x_peak-e0, y_peak))
            i+=1

    plt.figure(figsize=(12,6))
    plt.plot(E_centers, counts_plot, label="Spectrum", color='#1f77b4',
             linewidth=1.2, alpha=SPECTRUM_ALPHA_2, zorder=3)

    gauss_label_used=False
    for Ew_g, Yg_norm in gaussians_to_plot:
        if not gauss_label_used:
            plt.plot(Ew_g, Yg_norm, '--', color='#39FF14', alpha=SPECTRUM_ALPHA_2,
                     linewidth=2.0, zorder=10, label="Gaussian fit")
            gauss_label_used=True
        else:
            plt.plot(Ew_g, Yg_norm, '--', color='#39FF14', alpha=SPECTRUM_ALPHA_2,
                     linewidth=2.0, zorder=10)

    texts=[]
    if final_E:
        plt.scatter(final_E, final_Y_norm, color='red', s=16, zorder=12,
                    edgecolor='white', linewidth=0.5, label="Gaussian maxima")
        if ANNOTATE_PEAK_LABELS:
            for e,y in zip(final_E, final_Y_norm):
                texts.append(plt.text(e, y, f"{e:.1f} eV", fontsize=7, color='black',
                                      ha='center', zorder=14))

        # Anotar % SOLO en picos con entrada en CSV y ≠ 100
        for E_txt, mu_fit, A, sigma, c_est in peak_count_rows:
            pct, apply_pct = get_pct_for_peak_strict(E_txt)
            if apply_pct and np.isfinite(pct) and abs(pct - 100.0) > 1e-6:
                # colocar cerca del máximo más próximo al mu_fit
                j = int(np.argmin(np.abs(np.array(final_E) - mu_fit)))
                x_ = final_E[j]; y_ = final_Y_norm[j]
                label = f"{pct:.2f}% W"
                texts.append(plt.text(x_, y_, label, fontsize=8, color='black',
                                      ha='center', va='bottom', zorder=15))

        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        except Exception:
            pass

    plt.xlabel("Photon energy (eV)")
    if NORMALIZE_MODE_2.lower() in ('area','pdf','density'): plt.ylabel("Probability density (1/eV)")
    elif NORMALIZE_MODE_2.lower()=='max': plt.ylabel("Normalized counts (max=1)")
    else: plt.ylabel("Counts")
    plt.title("Single spectrum — peaks = maxima of fitted Gaussians (anotando %W solo si está en CSV)")
    plt.xlim(X_MIN_2, X_MAX_2); plt.legend(); plt.tight_layout()
    plt.savefig(IMG_NAME_2, dpi=200); plt.show()
    print(f"Saved image: {IMG_NAME_2}")

    if report_rows:
        print("\nResumen picos (TXT -> FIT):")
        print("  E_txt (eV)    E_fit (eV)    ΔE (eV) altura_norm")
        for E_txt, E_fit, dE, h in report_rows:
            print(f"  {E_txt:10.2f}  {E_fit:10.2f}  {dE:8.2f} {h:10.4f}")

    # Aquí aplicamos el porcentaje SOLO a los picos presentes en el CSV; el resto se trata como 100 %
    if peak_count_rows:
        print("\nCuentas por pico (solo término gaussiano, fondo excluido):")
        print("  E_txt (eV)    mu_fit (eV)        A(peak) sigma(eV)  cuentas_estimadas    %W_aplicado   cuentas_W")
        tungsten_counts_total = 0.0
        for E_txt, mu_fit, A, sigma, c_est in peak_count_rows:
            pct, apply_pct = get_pct_for_peak_strict(E_txt)
            factor = (pct/100.0) if (apply_pct and np.isfinite(pct)) else 1.0
            c_base = c_est if np.isfinite(c_est) else 0.0
            cW = factor * c_base
            tungsten_counts_total += cW
            A_show = f"{A:10.2f}" if np.isfinite(A) else " NaN   "
            sigma_show = f"{sigma:10.3f}" if np.isfinite(sigma) else "    NaN   "
            c_show = f"{c_est:14.2f}" if np.isfinite(c_est) else "      NaN   "
            pct_show = f"{pct:6.2f}%" if apply_pct and np.isfinite(pct) else " 100.00%"
            print(f"  {E_txt:10.2f}  {mu_fit:12.2f}  {A_show} {sigma_show}  {c_show}    {pct_show}   {cW:10.2f}")

        total_counts = float(np.sum(counts_raw))
        print(f"\nTotal cuentas en el rango [{X_MIN_2}, {X_MAX_2}] eV: {total_counts:.2f}")
        print(f"Suma ponderada por %W de los picos presentes en CSV: {tungsten_counts_total:.2f}")
        pct_global = 100.0 * tungsten_counts_total / total_counts if total_counts>0 else float('nan')
        print(f"Porcentaje global estimado de W en el espectro: {pct_global:.3f} %")
    else:
        print("\nNo se han podido estimar cuentas para los picos (sin ventanas válidas).")

# ===================== EJECUCIÓN =====================
if __name__ == "__main__":
    run_overlap_and_export()                 # 1ª figura + CSV con % por pico
    run_single_spectrum_with_annotations()   # 2ª figura + uso estricto del CSV
