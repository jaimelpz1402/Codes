import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# === 1. Cargar archivo real
data = np.load("/home/lopezgonza/Desktop/FEBIT/TES/W_measurement_3d_dataset_2.npy")

# === 2. Separar columnas
E_photon = data[2, :]
E_electron = data[1, :]

print(data.shape)

# === 3. Construir histograma 2D (solo desde 1000 eV)
xmin, xmax = 1000, np.max(E_electron)
ymin, ymax = 1000, np.max(E_photon)

x_bins = np.linspace(xmin, xmax, 500)
y_bins = np.linspace(ymin, ymax, 500)

hist2D, xedges, yedges = np.histogram2d(E_electron, E_photon, bins=[x_bins, y_bins])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# === 4. Potenciales de ionización (W)
potenciales_ionizacion = [
    7.86403, 16.37, 26.0, 38.2, 51.6, 64.77, 122.01, 141.2, 160.2, 179.0,
    208.9, 231.6, 258.3, 290.7, 325.3, 361.9, 387.9, 420.7, 462.1, 502.6,
    543.4, 594.5, 640.6, 685.6, 734.1, 784.4, 833.4, 881.4, 1132.2, 1180.0,
    1230.4, 1283.4, 1335.1, 1386.8, 1459.9, 1512.4, 1569.1, 1621.7, 1829.8,
    1882.9, 1940.6, 1994.8, 2149.1, 2210.0, 2354.5, 2414.1, 4057, 4180, 4309,
    4446, 4578, 4709, 4927, 5063, 5209, 5348, 5719, 5840, 5970, 6093, 6596,
    6735, 7000, 7130
]

# === 5. Plot
plt.figure(figsize=(10, 6))
plt.imshow(hist2D.T, origin='lower', extent=extent, aspect='auto',
           norm=mpl.colors.LogNorm(vmin=1), cmap='viridis')
plt.colorbar(label='Counts (log scale)')
plt.xlabel("Electron beam energy (eV)")
plt.ylabel("Photon energy (eV)")
plt.title("2D Spectrum with Tungsten Ionization Potentials")

# Limitar visualmente el rango desde 1000 eV
plt.xlim(1200, 8000)
plt.ylim(0, ymax)

contador = 0  # estado de ionización: empieza en W^0

for ip in potenciales_ionizacion:
    if ip < 1000:
        contador += 1
        continue  # ignorar líneas por debajo del umbral visual

    if contador % 10 == 0 and contador != 0:
        # Líneas destacadas (más intensas)
        plt.axvline(ip, color='black', linestyle=':', linewidth=1.2)
        plt.text(
            x=ip,
            y=300,  # etiqueta cerca de la parte inferior
            s=f"+{contador}",
            color='black',
            fontsize=7,
            ha='left',
            va='top'
        )
    else:
        # Líneas suaves
        plt.axvline(ip, color='gray', linestyle=':', linewidth=0.8)

    contador += 1

plt.tight_layout()
plt.savefig("/home/lopezgonza/Desktop/FEBIT/Figuras/W2plot.png")
plt.show()
