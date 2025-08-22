import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== PARÁMETROS AJUSTABLES =====================
# Rutas de entrada
PATH_DATA_A_NPY = "/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy" # Tungsteno (eventos): data[1,:]=E_beam, data[2,:]=E_photon
PATH_DATA_B_NPY = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run003_2025_01_30.npy" # Azufre (2 columnas/filas: counts y E_photon)

# Salida
OUTPUT_DIR = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selección de energía del cañón (se plotea SOLO esta)
TARGET_BEAM_ENERGY_EV = 4400.0
BEAM_TOLERANCE_EV     = 50.0

# Rango y binning del histograma (válido para TODO el flujo y la normalización)
X_MIN_EV = 700.0
X_MAX_EV = 4500.0
NUM_BINS = 2000

# Normalización:
#   'max'   -> escala a máximo 1 en el rango;
#   'area'  -> densidad (área=1 en el rango);
#   'none'  -> cuentas crudas;
#   'peak'  -> densidad con área=1 en una ventana [PEAK_CENTER_EV ± PEAK_WIDTH_EV/2]
NORMALIZE_MODE = "peak"   # "max" | "area" | "none" | "peak"

# Parámetros del modo 'peak'
PEAK_CENTER_EV = 3368.8   # Centro del pico de referencia
PEAK_WIDTH_EV  = 10.0     # Ancho total de la ventana en eV (no FWHM; ventana rectangular)

# Estética
SPECTRUM_ALPHA = 0.8
LINEWIDTH = 1.6
DPI_SAVE = 200
# ================================================================


# -------------------- CARGA DE DATOS --------------------
def load_spectrum_A(path_npy):
    """A: eventos de Tungsteno. Devuelve (E_beam, E_photon)."""
    data = np.load(path_npy)
    E_beam   = data[1, :].astype(float)
    E_photon = data[2, :].astype(float)
    return E_beam, E_photon


def load_spectrum_B_raw(path_npy):
    """
    B: Azufre binned en .npy con dos columnas/filas (orden desconocido).
    Devuelve (v1, v2) tal cual, sin decidir qué es energía o cuentas aún.
    """
    arr = np.load(path_npy)
    if arr.ndim != 2:
        raise ValueError("El .npy del azufre debe ser 2D con dos columnas o dos filas.")
    if arr.shape[1] == 2:
        v1 = arr[:, 0].astype(float)
        v2 = arr[:, 1].astype(float)
    elif arr.shape[0] == 2:
        v1 = arr[0, :].astype(float)
        v2 = arr[1, :].astype(float)
    else:
        raise ValueError("Se esperan exactamente dos columnas (N,2) o dos filas (2,N).")
    return v1, v2


# -------------------- HISTOGRAMA DE W --------------------
def histogram_A_for_beam(E_beam, E_photon, beam_energy, tol_ev, x_min, x_max, num_bins):
    """
    Histograma de W (Tungsteno) EN EL RANGO [x_min, x_max] y bins uniformes.
    Devuelve (counts_A, centers_A, edges_A).
    """
    mask_beam = np.abs(E_beam - beam_energy) <= tol_ev
    eph = E_photon[mask_beam]
    if eph.size == 0:
        return None, None, None
    counts, edges = np.histogram(eph, bins=num_bins, range=(x_min, x_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centers, edges


# -------------------- UTILIDADES DE BINS --------------------
def centers_to_edges_clamped(centers, x_min, x_max):
    """
    Construye bordes (edges) desde centros y los recorta a [x_min, x_max].
    """
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
    """
    Rebin proporcional por solape de intervalos.
    Suma las cuentas de cada bin de origen en los bins de destino, conservando el total.
    """
    src_edges = np.asarray(src_edges, dtype=float)
    src_counts = np.asarray(src_counts, dtype=float)
    tgt_edges = np.asarray(tgt_edges, dtype=float)

    nS = len(src_counts)
    nT = len(tgt_edges) - 1
    out = np.zeros(nT, dtype=float)

    i = j = 0
    while i < nS and j < nT:
        Ls, Rs = src_edges[i], src_edges[i+1]
        Lt, Rt = tgt_edges[j], tgt_edges[j+1]

        if Rs <= Lt:
            i += 1
            continue
        if Rt <= Ls:
            j += 1
            continue

        overlap = min(Rs, Rt) - max(Ls, Lt)
        if overlap > 0:
            width_src = max(Rs - Ls, 1e-12)
            out[j] += src_counts[i] * (overlap / width_src)

        if Rs <= Rt:
            i += 1
        else:
            j += 1

    return out


# -------------------- NORMALIZACIÓN (SOLO EN EL RANGO) --------------------
def normalization_factor(values, mode, bin_width=None):
    """
    Devuelve el factor por el que multiplicar 'values' para normalizar EN EL RANGO
    actualmente representado por esos 'values'.
      - 'max'  : 1 / max(values)
      - 'area' : 1 / (sum(values) * bin_width)   -> densidad (área total = 1 en el rango)
      - 'none' : 1.0
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0
    mode = (mode or 'none').lower()
    if mode == 'none':
        return 1.0
    if mode in ('area', 'pdf', 'density'):
        total = float(np.sum(values))
        if bin_width is None:
            return 1.0 if total <= 0 else 1.0 / total
        total *= float(bin_width)
        return 1.0 if total <= 0 else 1.0 / total
    if mode == 'max':
        m = float(np.max(values))
        return 1.0 if m <= 0 else 1.0 / m
    return 1.0


def normalization_factor_peak_from_edges(values, edges, bin_width, peak_center, peak_width):
    """
    Factor para escalar por el ÁREA dentro de la ventana del pico:
    ventana = [peak_center - peak_width/2, peak_center + peak_width/2].

    Integra por solape proporcional usando los 'edges' reales de los bins.
    Devuelve 1/area_ventana, usando densidad (multiplica por bin_width).
    Si el área es <= 0, devuelve 1.0 (no cambia la curva).
    """
    values = np.asarray(values, dtype=float)
    edges  = np.asarray(edges,  dtype=float)
    if values.size == 0 or edges.size < 2 or bin_width is None or peak_width <= 0:
        return 1.0

    half = 0.5 * float(peak_width)
    L = float(peak_center) - half
    R = float(peak_center) + half

    # Rebin a la ventana [L, R] para obtener las cuentas en la ventana con solape parcial
    win_edges = np.array([L, R], dtype=float)
    counts_in_window = float(np.sum(rebin_counts_to_edges(edges, values, win_edges)))

    # Convertimos a área (densidad) multiplicando por el ancho de bin
    area = counts_in_window * float(bin_width)
    return 1.0 if area <= 0.0 else 1.0 / area


# -------------------- LÓGICA DE AZUFRE (detección + rebin) --------------------
def choose_energy_counts_and_rebin(v1, v2, tgt_edges, x_min, x_max):
    """
    Prueba ambas orientaciones posibles de B:
      (E=v1, C=v2) y (E=v2, C=v1).
    Para cada una:
      - Ordena por E
      - Filtra centros dentro de [x_min, x_max]
      - Construye edges_B recortados al rango
      - Rebina a tgt_edges
    Devuelve los counts de B ya rebinados a los bins de W y un string con la orientación elegida.
    """
    def attempt(E, C):
        idx = np.argsort(E)
        E_sorted = np.asarray(E)[idx]
        C_sorted = np.asarray(C, dtype=float)[idx]
        mask = (E_sorted >= x_min) & (E_sorted <= x_max)
        E_in = E_sorted[mask]
        C_in = C_sorted[mask]
        if E_in.size == 0:
            return np.zeros(len(tgt_edges) - 1, dtype=float), 0.0
        edges_B = centers_to_edges_clamped(E_in, x_min, x_max)
        rebinned = rebin_counts_to_edges(edges_B, C_in, tgt_edges)
        return rebinned, float(np.sum(rebinned))

    r1, s1 = attempt(v1, v2)  # Opción 1: v1=E, v2=C
    r2, s2 = attempt(v2, v1)  # Opción 2: v2=E, v1=C

    if s1 > s2:
        return r1, "E=v1, C=v2"
    else:
        return r2, "E=v2, C=v1"


# -------------------- PLOT PRINCIPAL --------------------
def plot_single_energy(beam_energy, Eb, Eph, v1_B, v2_B, outdir):
    # Histograma de W en el rango y bins objetivo (los datos ya quedan limitados al rango)
    counts_A, centers_A, edges_A = histogram_A_for_beam(
        Eb, Eph, beam_energy, BEAM_TOLERANCE_EV, X_MIN_EV, X_MAX_EV, NUM_BINS
    )
    if counts_A is None:
        print(f"[INFO] No hay eventos para E_beam ~ {beam_energy:.1f} eV (±{BEAM_TOLERANCE_EV} eV).")
        return

    # Bin width uniforme (por construcción)
    binw_A = float(edges_A[1] - edges_A[0]) if len(edges_A) > 1 else None

    # Azufre: detectar orientación y rebinar a los bins de W (también solo en el rango)
    counts_B_on_A, orientation = choose_energy_counts_and_rebin(
        v1_B, v2_B, edges_A, X_MIN_EV, X_MAX_EV
    )

    # ---------------- Normalización ----------------
    mode = NORMALIZE_MODE.lower()
    if mode == "peak":
        # Normalización por área en la ventana del pico usando solape proporcional
        sA = normalization_factor_peak_from_edges(counts_A, edges_A, binw_A, PEAK_CENTER_EV, PEAK_WIDTH_EV)
        sB = normalization_factor_peak_from_edges(counts_B_on_A, edges_A, binw_A, PEAK_CENTER_EV, PEAK_WIDTH_EV)
        A_y = counts_A * sA
        B_y = counts_B_on_A * sB
        y_label = "Density (1/eV) [peak window area = 1]"
    else:
        sA = normalization_factor(counts_A, mode, bin_width=binw_A)
        sB = normalization_factor(counts_B_on_A, mode, bin_width=binw_A)
        A_y = counts_A * sA
        B_y = counts_B_on_A * sB
        if mode in ('area', 'pdf', 'density'):
            y_label = "Density (1/eV)"
        elif mode == 'max':
            y_label = "Normalized counts (max=1 in range)"
        else:
            y_label = "Counts"

    # ---------------------------- Plot ---------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(centers_A, A_y, label=f"W @ E_beam {beam_energy:.1f} eV (±{BEAM_TOLERANCE_EV:.0f})",
             linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA)
    plt.plot(centers_A, B_y, label=f"S (rebinned to W bins) [{orientation}]",
             linewidth=LINEWIDTH, alpha=SPECTRUM_ALPHA)

    # Marcar ventana del pico si procede
    if mode == "peak":
        half = 0.5 * PEAK_WIDTH_EV
        L = max(PEAK_CENTER_EV - half, X_MIN_EV)
        R = min(PEAK_CENTER_EV + half, X_MAX_EV)
        plt.axvspan(L, R, alpha=0.15, label=f"Peak window [{PEAK_CENTER_EV:.1f}±{half:.1f}] eV")

    plt.xlim(X_MIN_EV, X_MAX_EV)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel(y_label)
    plt.title("Tungsten (W) vs Sulfur (S)")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(
        outdir,
f"comparacion_W_vs_S_{beam_energy:.1f}eV_{NORMALIZE_MODE}_"
        f"{int(X_MIN_EV)}-{int(X_MAX_EV)}.png"
    )
    plt.savefig(fname, dpi=DPI_SAVE)
    print(f"[OK] Guardado: {fname}")
    plt.show()


def main():
    # Carga
    E_beam, E_photon = load_spectrum_A(PATH_DATA_A_NPY)
    v1_B, v2_B = load_spectrum_B_raw(PATH_DATA_B_NPY)

    # Plotea SOLO la energía seleccionada, todo limitado al rango y con bins idénticos
    plot_single_energy(TARGET_BEAM_ENERGY_EV, E_beam, E_photon, v1_B, v2_B, OUTPUT_DIR)


if __name__ == "__main__":
    main()
