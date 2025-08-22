import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ===================== PARÁMETROS BÁSICOS =====================
PATH_DATA_S_NPY = "/home/lopezgonza/Desktop/FEBIT/TES/Compiled_Ssteady_run002_40pix.npy"
OUTPUT_DIR = "/home/lopezgonza/Desktop/FEBIT/Figuras/Comparaciones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_MIN_EV  = 700.0
X_MAX_EV  = 4500.0
NUM_BINS  = 2000

# ÚNICO PARÁMETRO DEL DETECTOR (0–100)
PEAK_PROMINENCE_PERCENT = 0.4
# =============================================================

def load_sulfur_raw(path_npy):
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

def choose_energy_counts_and_rebin_to_uniform(v1, v2, x_min, x_max, num_bins):
    tgt_edges = np.linspace(x_min, x_max, num_bins + 1)
    def attempt(E, C):
        idx = np.argsort(E)
        E_sorted = np.asarray(E)[idx]
        C_sorted = np.asarray(C)[idx]
        mask = (E_sorted >= x_min) & (E_sorted <= x_max)
        E_in = E_sorted[mask]
        C_in = C_sorted[mask]
        if E_in.size == 0:
            return np.zeros(num_bins, dtype=float), 0.0
        edges_src = centers_to_edges_clamped(E_in, x_min, x_max)
        rebinned = rebin_counts_to_edges(edges_src, C_in, tgt_edges)
        return rebinned, float(np.sum(rebinned))

    r1, s1 = attempt(v1, v2)  # opción 1: v1=E, v2=C
    r2, s2 = attempt(v2, v1)  # opción 2: v2=E, v1=C

    counts = r1 if s1 >= s2 else r2
    centers = 0.5 * (tgt_edges[:-1] + tgt_edges[1:])
    binw = float(tgt_edges[1] - tgt_edges[0])
    orientation = "E=v1, C=v2" if s1 >= s2 else "E=v2, C=v1"
    return counts, centers, binw, orientation

def find_peaks_percent(counts_raw, prominence_percent):
    if counts_raw.size == 0:
        return np.array([], dtype=int), {}
    max_counts = float(np.max(counts_raw))
    if max_counts <= 0:
        return np.array([], dtype=int), {}
    prom_abs = (prominence_percent / 100.0) * max_counts
    idx, props = find_peaks(counts_raw, prominence=prom_abs)
    return idx, props

def plot_sulfur_only(v1_S, v2_S, outdir):
    counts_S_raw, centers_S, binw, orientation = choose_energy_counts_and_rebin_to_uniform(
        v1_S, v2_S, X_MIN_EV, X_MAX_EV, NUM_BINS
    )

    pk_idx, pk_props = find_peaks_percent(counts_S_raw, PEAK_PROMINENCE_PERCENT)

    peak_E = centers_S[pk_idx] if pk_idx.size > 0 else np.array([])
    peak_y = counts_S_raw[pk_idx] if pk_idx.size > 0 else np.array([])

    plt.figure(figsize=(12, 6))
    plt.plot(centers_S, counts_S_raw, label=f"Sulfur (S) [{orientation}]")
    if pk_idx.size > 0:
        plt.scatter(peak_E, peak_y, s=28, zorder=3, label="Peaks")
        for e, yv in zip(peak_E, peak_y):
            plt.annotate(f"{e:.1f} eV", xy=(e, yv),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)
    plt.xlim(X_MIN_EV, X_MAX_EV)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Counts")
    plt.title(f"Sulfur (S) spectrum | peaks: {len(peak_E)} | prom≥{PEAK_PROMINENCE_PERCENT:.1f}% del máximo")
    plt.legend()
    plt.tight_layout()

    base = f"S_spectrum_counts_{int(X_MIN_EV)}-{int(X_MAX_EV)}"
    png_path = os.path.join(outdir, base + ".png")
    plt.savefig(png_path, dpi=200)
    plt.show()
    print(f"[OK] Guardado plot: {png_path}")
    if counts_S_raw.size > 0:
        print(f"Max counts: {counts_S_raw.max():.1f} | Prominence abs usada: {counts_S_raw.max()*PEAK_PROMINENCE_PERCENT/100:.1f}")

def main():
    v1_S, v2_S = load_sulfur_raw(PATH_DATA_S_NPY)
    plot_sulfur_only(v1_S, v2_S, OUTPUT_DIR)

if __name__ == "__main__":
    main()
