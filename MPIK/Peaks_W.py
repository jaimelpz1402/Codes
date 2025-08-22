import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ===================== PARÁMETROS AJUSTABLES =====================
# Ruta de entrada: ARRAY de eventos con 3 columnas (o 3 filas):
# [E_beam (eV), E_photon (eV), t (s o a.u.)]  — el orden lo defines abajo en COL_*
PATH_DATA_3COL_NPY = "/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy"

# Salida
OUTPUT_DIR = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Selección por ENERGÍA DEL HAZ (e-beam) ----
# Filtro central (target ± tolerancia). Alternativamente, puedes ignorar el target y usar E_BEAM_MIN/MAX directamente.
E_BEAM_TARGET_EV = 4400.0
E_BEAM_TOL_EV    = 50.0     # se aceptan eventos con E_beam en [target - tol, target + tol]

# Si prefieres límites explícitos, descomenta y usalos en vez del target:
# E_BEAM_MIN_EV = 2490.0
# E_BEAM_MAX_EV = 2510.0

# ---- Rango y binning del histograma de energía de FOTONES ----
X_MIN_EV = 700.0
X_MAX_EV = 4500.0
NUM_BINS = 3000  # lo dejas a mano (p.ej., 3000) según prefieras

# Normalización del PLOT: 'max' -> máximo=1; 'area' -> densidad (área=1 en el rango); 'none' -> cuentas crudas
NORMALIZE_MODE = "area"   # "max" | "area" | "none"

# Estética
SPECTRUM_ALPHA = 0.8
LINEWIDTH = 1.6
DPI_SAVE = 200
FIGSIZE = (12, 6)

# ----------- DETECTOR DE PICOS (sobre CUENTAS CRUDAS del histograma) -----------
PEAK_PROMINENCE_COUNTS = 40.0     # prominencia mínima en CUENTAS
PEAK_MIN_DISTANCE_EV   = 1.0      # distancia mínima entre picos en eV (se pasa a bins)
PEAK_MIN_HEIGHT_COUNTS = None     # None para no exigir altura mínima
SAVE_PEAKS_TXT         = True
# ==============================================================================

# ---- Mapeo de columnas (si tu .npy es 3xN en vez de Nx3, el código lo detecta y transpone) ----
COL_E_BEAM   = 1
COL_E_PHOTON = 2
COL_TIME     = 0
# -----------------------------------------------------------------------------------------------


# -------------------- UTILIDADES --------------------
def normalization_factor(values, mode, bin_width=None):
    """
    Devuelve el factor multiplicativo para normalizar EN EL RANGO:
      - 'max'  : 1 / max(values)
      - 'area' : 1 / (sum(values) * bin_width)   (a densidad, área total = 1)
      - 'none' : 1.0
    """
    if values.size == 0:
        return 1.0
    mode = (mode or 'none').lower()
    if mode == 'none':
        return 1.0
    if mode in ('area', 'pdf', 'density'):
        total = float(np.sum(values))
        if bin_width is not None:
            total *= float(bin_width)
        return (1.0 / total) if total > 0 else 1.0
    if mode == 'max':
        m = float(np.max(values))
        return (1.0 / m) if m > 0 else 1.0
    return 1.0


def load_events_3cols(path, col_e_beam=0, col_e_photon=1, col_time=2):
    """
    Carga un .npy con eventos de 3 columnas o 3 filas.
    Devuelve tres arrays 1D: E_beam, E_photon, t
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError("El .npy debe ser 2D: Nx3 o 3xN.")
    # Transponer si viene como 3xN
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.shape[1] != 3:
        raise ValueError(f"Se esperaban 3 columnas, encontrado shape={arr.shape}.")
    E_beam   = arr[:, col_e_beam  ].astype(float)
    E_photon = arr[:, col_e_photon].astype(float)
    t        = arr[:, col_time    ].astype(float)
    return E_beam, E_photon, t


def filter_by_e_beam(E_beam, E_photon, t,
                     target=None, tol=None,
                     explicit_min=None, explicit_max=None):
    """
    Filtra por energía de haz:
      - Si se da target y tol, usa [target - tol, target + tol]
      - Si se dan explicit_min/explicit_max, usa esos límites
    Devuelve E_beam_f, E_photon_f, t_f
    """
    if explicit_min is not None and explicit_max is not None:
        emin, emax = float(explicit_min), float(explicit_max)
    elif target is not None and tol is not None:
        emin, emax = float(target - tol), float(target + tol)
    else:
        raise ValueError("Debes especificar (target, tol) o (explicit_min, explicit_max) para filtrar E_beam.")

    mask = (E_beam >= emin) & (E_beam <= emax)
    return E_beam[mask], E_photon[mask], t[mask], (emin, emax)


def histogram_photons(E_photon, x_min, x_max, num_bins):
    """
    Histograma simple de E_photon dentro del rango [x_min, x_max].
    Devuelve counts (crudas), centers, bin_width.
    """
    edges = np.linspace(x_min, x_max, num_bins + 1)
    counts, edges = np.histogram(E_photon, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = float(edges[1] - edges[0])
    return counts.astype(float), centers, binw


def find_peaks_on_counts(counts_raw, bin_width_ev, prominence_counts, min_distance_ev, min_height_counts=None):
    """
    Aplica scipy.signal.find_peaks sobre CUENTAS CRUDAS del histograma.
    """
    if counts_raw.size == 0:
        return np.array([], dtype=int), {}
    distance_bins = max(int(round(min_distance_ev / bin_width_ev)), 1)
    kwargs = {
        "prominence": float(prominence_counts),
        "distance": distance_bins
    }
    if min_height_counts is not None:
        kwargs["height"] = float(min_height_counts)
    idx, props = find_peaks(counts_raw, **kwargs)
    return idx, props


# -------------------- PLOT + GUARDADO --------------------
def plot_filtered_spectrum(E_beam_f, E_photon_f, t_f, ebeam_range, outdir):
    # Histograma de fotones en el rango seleccionado
    counts_raw, centers, binw = histogram_photons(E_photon_f, X_MIN_EV, X_MAX_EV, NUM_BINS)

    # Detección de picos sobre CUENTAS CRUDAS
    pk_idx, pk_props = find_peaks_on_counts(
        counts_raw,
        binw,
        prominence_counts=PEAK_PROMINENCE_COUNTS,
        min_distance_ev=PEAK_MIN_DISTANCE_EV,
        min_height_counts=PEAK_MIN_HEIGHT_COUNTS
    )

    # Serie para el plot según NORMALIZE_MODE (la detección ya se hizo con counts_raw)
    s = normalization_factor(counts_raw, NORMALIZE_MODE, bin_width=binw)
    y_plot = counts_raw * s

    # Datos de picos para anotar (en unidades del plot)
    peak_E = centers[pk_idx] if pk_idx.size > 0 else np.array([])
    peak_y = y_plot[pk_idx]  if pk_idx.size > 0 else np.array([])
    peak_prom_plot = (pk_props.get("prominences", np.array([])) * s) if pk_idx.size > 0 else np.array([])

    # Plot
    plt.figure(figsize=FIGSIZE)
    ebeam_label = f"E_beam ∈ [{ebeam_range[0]:.1f}, {ebeam_range[1]:.1f}] eV"
    plt.plot(centers, y_plot, label=f"Spectrum (filtered by {ebeam_label})",
             linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA)

    if pk_idx.size > 0:
        plt.scatter(peak_E, peak_y, s=28, zorder=3, label="Peaks")
        for e, yv in zip(peak_E, peak_y):
            plt.annotate(f"{e:.1f} eV",
                         xy=(e, yv),
                         xytext=(0, 8),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    plt.xlim(X_MIN_EV, X_MAX_EV)
    plt.xlabel("Photon energy (eV)")
    if NORMALIZE_MODE.lower() in ('area', 'pdf', 'density'):
        plt.ylabel("Density (1/eV)")
    elif NORMALIZE_MODE.lower() == 'max':
        plt.ylabel("Normalized counts")
    else:
        plt.ylabel("Counts")

    title_pk = f" | peaks: {len(peak_E)}" if peak_E.size else ""
    plt.title(f"Photon spectrum filtered by e-beam{title_pk}")
    plt.legend()
    plt.tight_layout()

    # Guardado
    base = (f"spectrum_Ebeam_{ebeam_range[0]:.1f}-{ebeam_range[1]:.1f}_"
f"{NORMALIZE_MODE}_{int(X_MIN_EV)}-{int(X_MAX_EV)}_bins{NUM_BINS}")
    png_path = os.path.join(outdir, base + ".png")
    plt.savefig(png_path, dpi=DPI_SAVE)
    print(f"[OK] Guardado plot: {png_path}")
    plt.show()

    # Guardar lista de picos
    if SAVE_PEAKS_TXT and peak_E.size:
        txt_path = os.path.join(outdir, base + "_peaks.txt")
        header = ("# Peaks found on spectrum filtered by e-beam (raw-counts detection)\n"
                  f"# EBEAM_RANGE: [{ebeam_range[0]:.6f},{ebeam_range[1]:.6f}] eV\n"
                  f"# PHOTON_RANGE: [{X_MIN_EV},{X_MAX_EV}] eV | NUM_BINS={NUM_BINS} | binw={binw:.6f} eV\n"
                  f"# PARAMS: prominence≥{PEAK_PROMINENCE_COUNTS} counts, "
                  f"distance≥{PEAK_MIN_DISTANCE_EV} eV, "
                  f"min_height={'None' if PEAK_MIN_HEIGHT_COUNTS is None else PEAK_MIN_HEIGHT_COUNTS} counts\n"
                  "# Columns: E_peak_eV  height_plot prominence_plot  left_base_eV  right_base_eV\n")
        left_bases  = centers[pk_props["left_bases"]] if "left_bases" in pk_props else np.full_like(peak_E, np.nan)
        right_bases = centers[pk_props["right_bases"]] if "right_bases" in pk_props else np.full_like(peak_E, np.nan)
        data = np.column_stack([peak_E,
                                peak_y,
                                peak_prom_plot,
                                left_bases,
                                right_bases])
        np.savetxt(txt_path, data, header=header)
        print(f"[OK] Guardado picos: {txt_path}")


def main():
    # Carga
    E_beam, E_photon, t = load_events_3cols(
        PATH_DATA_3COL_NPY,
        col_e_beam=COL_E_BEAM,
        col_e_photon=COL_E_PHOTON,
        col_time=COL_TIME
    )

    # Filtro por E_beam (elige una de las dos modalidades)
    # 1) Target ± tolerancia:
    E_beam_f, E_photon_f, t_f, ebeam_range = filter_by_e_beam(
        E_beam, E_photon, t,
        target=E_BEAM_TARGET_EV, tol=E_BEAM_TOL_EV
    )
    # 2) Límites explícitos:
    # E_beam_f, E_photon_f, t_f, ebeam_range = filter_by_e_beam(
    #     E_beam, E_photon, t,
    #     explicit_min=E_BEAM_MIN_EV, explicit_max=E_BEAM_MAX_EV
    # )

    print(f"[INFO] Eventos totales: {len(E_beam)} | Tras filtro E_beam: {len(E_beam_f)}")
    if len(E_beam_f) == 0:
        print("[ADVERTENCIA] No hay eventos tras el filtro de E_beam. Revisa target/tolerancia o límites.")
        return

    # Plot y guardado
    plot_filtered_spectrum(E_beam_f, E_photon_f, t_f, ebeam_range, OUTPUT_DIR)


if __name__ == "__main__":
    main()
