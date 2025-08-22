import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares
import matplotlib.patheffects as pe
import os

# ===================== PARÁMETROS AJUSTABLES =====================
ION_START = 42   # Ion mínimo a graficar (p. ej., 43 = W+43)
ION_END   = 62   # Ion máximo a graficar (p. ej., 54 = W+54)

N_BINS = 2000    # Número de bins del histograma para el rango [X_MIN, X_MAX]

WINDOW_FACTOR = 1.3            # margen por lado = WINDOW_FACTOR * (resolución FWHM estimada)
DISTANCE_OVERRIDE_EV = None    # si no es None, fuerza la distancia de agrupación en eV; si None, usa FWHM medida

# === Normalización para ploteo: 'area' | 'max' | 'none'
NORMALIZE_MODE = 'max'
SPECTRUM_ALPHA = 0.5

# ----------------- MEDICIÓN DE RESOLUCIÓN (FWHM) -----------------
RES_SEARCH_WINDOW_EV = 40.0
RES_MIN_PROM_FRAC_OF_MAX = 0.10
RES_DISTANCE_EV = 8.0
RES_MIN_ISOLATION_DROP = 0.50
RES_SNR_THRESHOLD = 5.0
RES_SUMMARY_STAT = 'median'
# -----------------------------------------------------------------

# === Rango de energía para computar y plotear
X_MIN = 700.0
X_MAX = 4500.0

# === 0. Energías de ionización (cortes del haz)
ruta_txt = "/home/lopezgonza/Desktop/FEBIT/ionization_energies.txt"
with open(ruta_txt, 'r') as f:
    ionization_energies = np.array([float(line.strip()) for line in f if line.strip()])

# === 1. Cargar dataset completo
data = np.load("/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy")
E_beam = data[1, :]
E_photon = data[2, :]

# === Salida
img_dir = "/home/lopezgonza/Desktop/FEBIT/Figuras/Prueba/Datos2"
os.makedirs(img_dir, exist_ok=True)

# ===================== FUNCIONES AUXILIARES =====================
def gauss_lin(E, p):
    b0, b1, A, mu, sigma = p
    sigma = max(float(sigma), 1e-12)
    return b0 + b1*E + A * np.exp(-0.5*((E - mu)/sigma)**2)

def residuals_gauss_lin(p, E, Y):
    return gauss_lin(E, p) - Y

def fit_single_gaussian_window(Ew, Yw, mu0=None, sigma0=None):
    if mu0 is None:
        w = np.clip(Yw - np.percentile(Yw, 20), 0, None)
        sw = np.sum(w)
        mu0 = np.sum(w*Ew)/sw if sw > 0 else Ew[np.argmax(Yw)]
    if sigma0 is None:
        sigma0 = max((Ew.max()-Ew.min())/8.0, max(np.median(np.diff(Ew)), 1e-12)*2)

    A_ls = np.vstack([Ew, np.ones_like(Ew)]).T
    b1_init, b0_init = np.linalg.lstsq(A_ls, Yw, rcond=None)[0]
    A_init = max(Yw.max() - (b0_init + b1_init*mu0), (Yw.max()-np.median(Yw))*0.5)

    p0 = np.array([b0_init, b1_init, A_init, mu0, sigma0], dtype=float)
    dEw = max(np.median(np.diff(Ew)), 1e-12)
    bounds = (
        np.array([-np.inf, -np.inf, 0.0, Ew.min(), 0.5*dEw]),
        np.array([ np.inf,  np.inf, np.inf, Ew.max(), (Ew.max()-Ew.min())*2.0])
    )

    res = least_squares(
        residuals_gauss_lin, p0, args=(Ew, Yw),
        loss='soft_l1', f_scale=1.0, bounds=bounds, max_nfev=20000
    )
    p = res.x
    return {
        'success': res.success,
        'params': {'b0': p[0], 'b1': p[1], 'A': p[2], 'mu': p[3], 'sigma': p[4]},
        'model_curve': gauss_lin(Ew, p),
        'Ew': Ew
    }

def normalization_factor(counts, mode, bin_width=None):
    if counts.size == 0:
        return 1.0
    mode = (mode or 'none').lower()
    if mode == 'none':
        return 1.0
    if mode in ('area', 'pdf', 'density'):
        total = float(np.sum(counts))
        if bin_width is not None:
            total *= float(bin_width)
        return (1.0 / total) if total > 0 else 1.0
    if mode == 'max':
        m = float(np.max(counts))
        return (1.0 / m) if m > 0 else 1.0
    return 1.0

# ---------- RESOLUCIÓN no paramétrica ----------
def _local_background(x, y, mu_guess, window):
    mask = (x > mu_guess - window) & (x < mu_guess + window)
    if not np.any(mask):
        return 0.0, 0.0
    xv, yv = x[mask], y[mask]
    core = (xv > mu_guess - window/4) & (xv < mu_guess + window/4)
    xb, yb = xv[~core], yv[~core]
    if len(xb) < 3:
        xb, yb = xv, yv
    A = np.vstack([np.ones_like(xb), (xb - mu_guess)]).T
    B0, B1 = np.linalg.lstsq(A, yb, rcond=None)[0]
    return float(B0), float(B1)

def _half_max_width(x, y):
    if len(x) < 4 or np.all(y <= 0):
        return np.nan, np.nan, False
    imax = int(np.argmax(y)); ymax = float(y[imax]); half = ymax/2.0

    i = imax
    while i > 0 and y[i] > half:
        i -= 1
    if i == 0:
        return np.nan, np.nan, False
    xL = x[i] + (x[i+1]-x[i])*(half - y[i])/(y[i+1]-y[i] + 1e-12)

    j = imax
    while j < len(y)-1 and y[j] > half:
        j += 1
    if j >= len(y)-1:
        return np.nan, np.nan, False
    xR = x[j-1] + (x[j]-x[j-1])*(half - y[j-1])/(y[j]-y[j-1] + 1e-12)
    return float(x[imax]), float(xR - xL), True

def measure_resolution_in_spectrum(E_centers, counts_raw):
    results = []
    if E_centers.size < 5 or counts_raw.size < 5 or np.max(counts_raw) <= 0:
        return results

    prom_abs = RES_MIN_PROM_FRAC_OF_MAX * float(np.max(counts_raw))
    mean_dE = np.median(np.diff(E_centers))
    min_distance_bins = max(1, int(RES_DISTANCE_EV / max(mean_dE, 1e-12)))
    peaks, _ = find_peaks(counts_raw, prominence=prom_abs, distance=min_distance_bins)
    if peaks.size == 0:
        return results

    med = np.median(counts_raw)
    mad = np.median(np.abs(counts_raw - med)) + 1e-9
    sigma_bg_global = 1.4826 * mad

    for p_idx in peaks:
        mu_guess = E_centers[p_idx]
        B0, B1 = _local_background(E_centers, counts_raw, mu_guess, window=RES_SEARCH_WINDOW_EV)
        mask = (E_centers > mu_guess - RES_SEARCH_WINDOW_EV) & (E_centers < mu_guess + RES_SEARCH_WINDOW_EV)
        x, y = E_centers[mask], counts_raw[mask]
        if x.size < 8:
            continue
        ycorr = y - (B0 + B1*(x - mu_guess)); ycorr[ycorr < 0] = 0.0

        core = (x > mu_guess - RES_SEARCH_WINDOW_EV/6) & (x < mu_guess + RES_SEARCH_WINDOW_EV/6)
        resid = y[~core] - (B0 + B1*(x[~core] - mu_guess))
        sigma_local = np.std(resid) if resid.size >= 5 else sigma_bg_global

        height = float(np.max(ycorr)); SNR = height/(sigma_local + 1e-9)
        if SNR < RES_SNR_THRESHOLD:
            continue

        imax_local = int(np.argmax(ycorr))
        left_valley = np.min(ycorr[:imax_local+1]) if imax_local > 0 else height
        right_valley = np.min(ycorr[imax_local:]) if imax_local < len(ycorr)-1 else height
        valley_level = max(left_valley, right_valley)
        if not (valley_level <= RES_MIN_ISOLATION_DROP * height):
            continue

        mu, fwhm, ok = _half_max_width(x, ycorr)
        if not ok or not np.isfinite(fwhm) or fwhm <= 0:
            continue

        results.append({
            'E_peak': mu,
            'FWHM_eV': fwhm,
            'rel_resolution': fwhm/mu,
            'SNR': SNR,
            'ok': True
        })

    results.sort(key=lambda r: r['E_peak'])
    return results

def summarize_resolution(results):
    if not results:
        return np.nan, 0
    vals = np.array([r['FWHM_eV'] for r in results], dtype=float)
    if RES_SUMMARY_STAT == 'mean':
        return float(np.mean(vals)), len(vals)
    if RES_SUMMARY_STAT == 'best':
        return float(np.min(vals)), len(vals)
    return float(np.median(vals)), len(vals)

def group_by_resolution(sorted_positions_e, res_eV):
    if len(sorted_positions_e) == 0 or not np.isfinite(res_eV) or res_eV <= 0:
        return [[j] for j in range(len(sorted_positions_e))]
    groups = [[0]]
    for j in range(1, len(sorted_positions_e)):
        if (sorted_positions_e[j] - sorted_positions_e[j-1]) < res_eV:
            groups[-1].append(j)
        else:
            groups.append([j])
    return groups

# ----------- NUEVO: Etiquetado exactamente encima del punto -----------
def place_labels_above(ax, xs, ys, fmt="{:.0f} eV", y_offset_pts=3):
    """
    Coloca etiquetas encima de cada punto (xs, ys), con un offset fijo en puntos.
    Esto evita depender de la escala del eje y del zoom.
    """
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(x),
            xy=(x, y),
            xytext=(0, y_offset_pts),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=20,
            clip_on=False,  # por si el texto queda muy cerca del borde
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white', alpha=0.9)]
        )

# =============================================================
# === 2. Iterar tríos consecutivos: (previo, actual, siguiente)
# state_curr = 30 + i  ->  i = state_curr - 30
i_min_req = max(ION_START - 30, 1)                             # al menos 1 porque usamos e_prev
i_max_req = min(ION_END - 30, len(ionization_energies) - 2)    # hasta len-2 porque usamos e_next

if i_min_req > i_max_req:
    print(f"[ADVERTENCIA] Rango de iones solicitado [{ION_START}, {ION_END}] no válido con los datos disponibles.")

for i in range(i_min_req, i_max_req + 1):
    e_prev = ionization_energies[i - 1]
    e_current = ionization_energies[i]
    e_next = ionization_energies[i + 1]

    cuts = [
        ((e_prev + e_current) / 2, abs(e_current - (e_prev + e_current) / 2)),   # Previous
        ((e_current + e_next) / 2,   abs(e_next   - (e_current + e_next) / 2))   # Posterior
    ]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    colors = ['#1f77b4', '#ff7f0e']
    colors_gauss = ['#39FF14', '#17becf']
    colors_peaks = ['red', 'purple']
    labels = ['Previous', 'Posterior']
    gauss_label_used = [False, False]
    peaks_label_used = [False, False]

    state_curr = 30 + i
    state_prev = state_curr - 1

    # para ajustar el límite superior y que no se recorten etiquetas
    y_max_plot = 0.0

    for k, (cut_value, tolerance) in enumerate(cuts):
        mask = np.abs(E_beam - cut_value) <= tolerance
        E_photon_cut = E_photon[mask]

        # Filtro duro en energía de fotón
        if E_photon_cut.size > 0:
            mask_x = (E_photon_cut >= X_MIN) & (E_photon_cut <= X_MAX)
            E_photon_cut = E_photon_cut[mask_x]
        if E_photon_cut.size == 0:
            continue

        # ====== HISTOGRAMA con N_BINS fijos en [X_MIN, X_MAX] ======
        counts_raw, edges = np.histogram(E_photon_cut, bins=N_BINS, range=(X_MIN, X_MAX))
        E_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_width = (X_MAX - X_MIN) / float(N_BINS)
        # ===========================================================

        # ---- Medir resolución en este espectro RAW (W+ion_state)
        res_list = measure_resolution_in_spectrum(E_centers, counts_raw)
        fwhm_summary, n_used = summarize_resolution(res_list)
        ion_state = (state_prev if k == 0 else state_curr)
        if np.isfinite(fwhm_summary):
            print(f"Resolución (FWHM) estimada para W+{ion_state}: {fwhm_summary:.2f} eV (n={n_used}, método=no-param)")
        else:
            print(f"Resolución (FWHM) estimada para W+{ion_state}: no disponible (sin picos aislados válidos)")

        # === Distancia de agrupación: FWHM medida o override manual
        if DISTANCE_OVERRIDE_EV is not None and np.isfinite(DISTANCE_OVERRIDE_EV) and DISTANCE_OVERRIDE_EV > 0:
            distance_for_grouping_eV = float(DISTANCE_OVERRIDE_EV)
        else:
            distance_for_grouping_eV = float(fwhm_summary) if np.isfinite(fwhm_summary) and fwhm_summary > 0 else 8.0

        # Normalización solo para visualizar
        s_norm = normalization_factor(counts_raw, NORMALIZE_MODE, bin_width=bin_width)
        counts_plot = counts_raw * s_norm
        if counts_plot.size > 0:
            y_max_plot = max(y_max_plot, float(np.max(counts_plot)))

        # Detección de picos sobre RAW
        if counts_raw.size > 0 and np.max(counts_raw) > 0:
            peak_idx = find_peaks(counts_raw, prominence=0.15 * np.max(counts_raw))[0]
        else:
            peak_idx = np.array([], dtype=int)

        peak_energies = E_centers[peak_idx]
        gaussians_to_plot = []
        final_E, final_Y_norm = [], []

        if peak_energies.size > 0:
            sort_idx = np.argsort(peak_energies)
            peak_energies_sorted = peak_energies[sort_idx]
            idxs_sorted = peak_idx[sort_idx]
            groups = group_by_resolution(peak_energies_sorted, distance_for_grouping_eV)
        else:
            groups = []

        for g in groups:
            if len(g) == 0:
                continue
            g_local_idx = np.array(g, dtype=int)
            g_int_idx = idxs_sorted[g_local_idx]
            g_Es = peak_energies_sorted[g_local_idx]

            # Ventana variable: margen por lado proporcional a la resolución
            margin_per_side = WINDOW_FACTOR * distance_for_grouping_eV

            # Bordes con margen
            E_left = g_Es.min() - margin_per_side
            E_right = g_Es.max() + margin_per_side

            # Recorte por puntos medios para evitar solapes con grupos vecinos
            a = g_local_idx[0]
            b = g_local_idx[-1]
            left_mid = 0.5 * (peak_energies_sorted[a] + peak_energies_sorted[a-1]) if a > 0 else X_MIN
            right_mid = 0.5 * (peak_energies_sorted[b] + peak_energies_sorted[b+1]) if b < len(peak_energies_sorted)-1 else X_MAX
            Emin = max(E_left, left_mid)
            Emax = min(E_right, right_mid)

            mask_win = (E_centers >= Emin) & (E_centers <= Emax)
            Ew = E_centers[mask_win]
            Yw_raw = counts_raw[mask_win]

            if Ew.size >= 5 and np.any(Yw_raw > 0):
                w = np.clip(Yw_raw - np.percentile(Yw_raw, 20), 0, None)
                sw = np.sum(w)
                mu0 = (np.sum(w*Ew)/sw) if sw > 0 else np.mean(g_Es)
                sigma0 = max((Emax - Emin)/8.0, max(np.median(np.diff(Ew)), 1e-12)*2)

                fit = fit_single_gaussian_window(Ew, Yw_raw, mu0=mu0, sigma0=sigma0)
                Yg_raw = fit['model_curve']; Ew_fit = fit['Ew']
                jmax = int(np.argmax(Yg_raw))
                x_peak = float(Ew_fit[jmax]); y_peak_norm = float(Yg_raw[jmax] * s_norm)
                final_E.append(x_peak); final_Y_norm.append(y_peak_norm)
                gaussians_to_plot.append((fit['Ew'], fit['model_curve'] * s_norm, k))
            else:
                for idx in g_int_idx:
                    final_E.append(float(E_centers[idx]))
                    final_Y_norm.append(float(counts_raw[idx] * s_norm))

        # Espectro y gaussianas
        ax.plot(
            E_centers, counts_plot,
            label=f"{labels[k]} spectrum: {cut_value:.1f} ± {tolerance:.1f} eV",
            color=colors[k], linewidth=1.2, alpha=SPECTRUM_ALPHA, zorder=3
        )

        for Ew_g, Yg_norm, kk in gaussians_to_plot:
            ax.plot(Ew_g, Yg_norm, '--', color=colors_gauss[kk], alpha=SPECTRUM_ALPHA,
                    linewidth=2.0, zorder=10, label=f"{labels[kk]} Gaussian" if not gauss_label_used[kk] else None)
            gauss_label_used[kk] = True
            if len(Yg_norm) > 0:
                y_max_plot = max(y_max_plot, float(np.max(Yg_norm)))

        if final_E:
            sc = ax.scatter(final_E, final_Y_norm, color=colors_peaks[k], s=12, zorder=12,
                            edgecolor='white', linewidth=0.4, label=f"{labels[k]} peaks" if not peaks_label_used[k] else None)
            peaks_label_used[k] = True
            if len(final_Y_norm) > 0:
                y_max_plot = max(y_max_plot, float(np.max(final_Y_norm)))

            # --- Etiquetas exactamente encima de cada punto ---
            place_labels_above(ax, final_E, final_Y_norm, fmt="{:.0f} eV", y_offset_pts=3)

    # Ejes y guardado (una figura por par de cortes Previous/Posterior)
    ax.set_xlabel("Photon energy (eV)")
    if NORMALIZE_MODE.lower() in ('area', 'pdf', 'density'):
        ax.set_ylabel("Probability density (1/eV)")
    elif NORMALIZE_MODE.lower() == 'max':
        ax.set_ylabel("Normalized counts (max=1)")
    else:
        ax.set_ylabel("Counts")

    ax.set_title(f"Beam energies sufficient for W+{state_prev} and W+{state_curr}")
    ax.set_xlim(X_MIN, X_MAX)

    # Asegurar que no se recorten etiquetas cerca del tope:
    if y_max_plot > 0:
        ax.set_ylim(bottom=0)
        y0, y1 = ax.get_ylim()
        y1_target = max(y1, y_max_plot * 1.08)  # +8% de margen
        if y1_target > y1:
            ax.set_ylim(y0, y1_target)

    ax.legend()
    plt.tight_layout()

    img_name = os.path.join(img_dir, f"W{state_prev}_{state_curr}_thresholds.png")
    plt.savefig(img_name, dpi=200)
    plt.show()
    print(f"[{i}/{len(ionization_energies)-2}] Saved image: {img_name}")
