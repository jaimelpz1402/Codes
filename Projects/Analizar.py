#Comienzo el programa incorporando las librerias necesarias para su correcto funcionamiento.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


iteraciones=100000 #Numero de iteraciones
directory = "C:/Users/jaime/Desktop/Compu/Problema1/Voluntario/Problema1"  #INGRESAR RUTA DONDE ESTÁN GUARADADOS
#LOS DATOS DESCARGADOS

#Función que carga valores
def load_planet_counts(directory, filename):
    # Cargar los conteos de planetas rocosos y gigantes gaseosos desde un archivo en el directorio especificado
    filename = f"{directory}/{filename}"
    with open(filename, 'r') as file:
        num_rocky_planets = int(file.readline().strip())
        num_gas_giants = int(file.readline().strip())
    return num_rocky_planets, num_gas_giants

N_Roc_Ant, N_Gas_Ant = load_planet_counts(directory, 'Numero_Planetas_Antiguos.txt')
N_Roc, N_Gas = load_planet_counts(directory, 'Numero_Planetas_Nuevos.txt')

print("Este fichero analizará los resultados obtenidos para la colisión de los planetesimales. Empezaré analizando el número")
print("de planetas antes y después de colisionar")
print()

print(f"El número de planetas rocosos antes de la colisión es {N_Roc_Ant}\n")
print(f"El número de planetas gaseosos antes la colisión es {N_Gas_Ant}\n")
print()
print(f"El número de planetas rocosos tras la colisión es {N_Roc}\n")
print(f"El número de planetas gaseosos tras la colisión es {N_Gas}\n")

#Función que carga valores
def cargar_datos(directory, filename, N, iteraciones):
    # Construir la ruta completa del archivo usando el directorio y el nombre del archivo
    filepath = f"{directory}/{filename}"
    
    # Inicializar un array tridimensional
    # Dimensiones: [Número de planetas, Coordenadas (2), Número de iteraciones]
    data_array = np.zeros((N, 2, iteraciones))
    
    with open(filepath, 'r') as file:
        for i in range(N):
            for k in range(iteraciones):
                while True:  # Esto manejará cualquier número de líneas vacías, incluso si hay más de una por error
                    linea = file.readline().strip()
                    if linea:  # Solo procesa líneas que no estén vacías
                        coords = list(map(float, linea.split(',')))
                        data_array[i, :, k] = coords
                        break  # Salir del bucle mientras después de procesar una línea válida

    return data_array

#Guardo valores de la posición
r_Roc = cargar_datos(directory, 'r_Roc_New.txt', N_Roc, iteraciones)
r_Gas = cargar_datos(directory, 'r_Gas_New.txt', N_Gas, iteraciones)

def load_values(directory, filename):
    filepath = f"{directory}/{filename}"
    with open(filepath, 'r') as file:
        # Usar list comprehension para crear una lista de flotantes
        value = [float(line.strip()) for line in file if line.strip()]
    
    # Convertir la lista a un array de NumPy y devolverlo
    return np.array(value)
ECal_Roc = load_values(directory, 'energy_calorifica_Roc.txt')
ECal_Gas = load_values(directory, 'energy_calorifica_Gas.txt')

#Desrescalo los valores de la energía
c=1.496*10**11  #m
Ms=2.167*10**31 #Kg
G=6.67384*10**(-11) #m^3 / Kg s^2
t=np.sqrt((c**3)/(G*Ms))

ECal_Roc=ECal_Roc*((c**2*Ms)/(t**2))
ECal_Gas=ECal_Gas*((c**2*Ms)/(t**2))

# Impresión amigable de los valores energéticos
print("A continuación se muestra la energía calorífica de cada planeta. Si no han chocado su energía calorífica será 0.")
print()
for i, energy in enumerate(ECal_Roc, 1):
    print(f"Planeta rocoso {i} tiene energía calorífica: {energy:.2e} J")
    print()
print()    
for i, energy in enumerate(ECal_Gas, 1):
    print(f"Planeta gaseoso {i} tiene energía calorífica: {energy:.2e} J")
    print()

#Función que calcula la distancia promedio de cada planeta a la estrella
def calcular_distancias_promedio(r):
    N = r.shape[0]  # Número de planetas
    iteraciones = r.shape[2]  # Número de iteraciones
    distancias_promedio = np.zeros(N)

    for i in range(N):
        suma_distancias = 0
        for k in range(iteraciones):
            x, y = r[i, 0, k], r[i, 1, k]
            distancia = np.sqrt(x**2 + y**2)
            suma_distancias += distancia
        distancias_promedio[i] = suma_distancias / iteraciones

    return distancias_promedio

# Calcular distancias promedio
distancias_promedio_rocosos = calcular_distancias_promedio(r_Roc)
distancias_promedio_gaseosos = calcular_distancias_promedio(r_Gas)

# Imprimir las distancias promedio con identificación detallada
print()
print("Ahora se estudiará la distancia promedio de los planetas a la estrella")
print()
for i, distancia in enumerate(distancias_promedio_rocosos):
    print(f"Distancia promedio del planeta rocoso {i} es: {distancia*c/(10**3):.2f} Km")
    print()

for i, distancia in enumerate(distancias_promedio_gaseosos):
    print(f"Distancia promedio del planeta gaseoso {i} es: {distancia*c/(10**3):.2f} Km")
    print()


Radio_Roc = load_values(directory, 'Radio_Roc.txt')
Radio_Gas = load_values(directory, 'Radio_Gas.txt')

#Desescalo los valores del radio
Radio_Roc=(Radio_Roc*c)/(10**3)
Radio_Gas=(Radio_Gas*c)/(10**3)


# Cálculo de los tamaños medios
average_size_roc = np.mean(Radio_Roc) 
average_size_gas = np.mean(Radio_Gas)
average_size_all = np.mean(np.concatenate([Radio_Roc, Radio_Gas]))

# Impresión de los tamaños medios
print()
print("Ahora se estudiará el tamaño medio de los planetas:")
print()
print(f"Tamaño medio de todos los planetesimales: {average_size_all:.3e} km")
print()
print(f"Tamaño medio de los planetesimales rocosos: {average_size_roc:.3e} km")
print()
print(f"Tamaño medio de los planetesimales gaseosos: {average_size_gas:.3e} km")
print()

#Función que clasifica los planetas según su radio.
def clasificar_planet(radi):
    categoria = {'pequeños': 0, 'medianos': 0, 'grandes': 0, 'muy grandes': 0}
    for radio in radi:
        valor=(1/1000)*((3*((3.038*10**28)/(N_Roc_Ant+N_Gas_Ant)))/(4*np.pi*3130))**(1/3)
        if radio <= valor:  
            categoria['pequeños'] += 1
        elif radio < 2*valor :  
            categoria['medianos'] += 1
        elif radio < 8*valor: 
            categoria['grandes'] += 1
        else:  
            categoria['muy grandes'] += 1
    return categoria
Radios_Planet = np.concatenate((Radio_Roc, Radio_Gas))
tamaño = clasificar_planet(Radios_Planet)

# Muestro los datos por pantalla
for size, count in tamaño.items():
    print(f"Número de planetas {size}: {count}")

print()

Periodo_Roc = load_values(directory, 'Periodos_Roc.txt')
Periodo_Gas = load_values(directory, 'Periodos_Gas.txt')

# Impresión de los períodos
print()
print("Períodos orbitales de los planetas:")
print()
for i, period in enumerate(Periodo_Roc, 1):
    print(f"Planeta rocoso número {i} tiene un período de {period:.2f} días")
    print()
print()    
for i, period in enumerate(Periodo_Gas, 1):
    print(f"Planeta gaseoso número {i} tiene un período de {period:.2f} días")
    print()

# Carga de valores de excentricidad para planetas rocosos y gaseosos
ex_Roc = load_values(directory, 'Ex_Roc.txt')
ex_Gas = load_values(directory, 'Ex_Gas.txt')

# Impresión de las excentricidades
print()
print("Información de las excentricidades de los planetas:")
print()
for i, ex in enumerate(ex_Roc, 1):
    print(f"Planeta rocoso número {i} tiene una excentricidad de {ex:.2f}")
    print()
print()
    
for i, ex in enumerate(ex_Gas, 1):
    print(f"Planeta gaseoso número {i} tiene una excentricidad de {ex:.2f}")
    print()

# Carga de valores de masa para planetas rocosos y gaseosos
Masa_Roc = load_values(directory, 'Masa_Roc.txt')
Masa_Gas = load_values(directory, 'Masa_Gas.txt')

# Normalización de las masas para el tamaño de los puntos
size_factor = 3.038e25
sizes_Roc = Masa_Roc / size_factor
sizes_Gas = Masa_Gas / size_factor

# Carga de valores de masa para planetas rocosos y gaseosos
Masa_Roc = load_values(directory, 'Masa_Roc.txt')
Masa_Gas = load_values(directory, 'Masa_Gas.txt')

# Normalización de las masas para el tamaño de los puntos
size_factor = 3.038e25
sizes_Roc = Masa_Roc / size_factor
sizes_Gas = Masa_Gas / size_factor

# Configuración de la visualización
x_min, x_max, y_min, y_max = -50, 50, -50, 50
interval = 0.01  # Intervalo entre frames en milisegundos
file_out = "animacion_orbitas"  # Archivo de salida para la animación
save_to_file = False  # Controla si se guarda la animación como un archivo
dpi = 150  # Resolución para el archivo de salida

# Crear figura y eje para la animación
fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')

# Inicializar puntos y líneas para cada planetesimal
points_Roc = [ax.plot([], [], 'o', color='red', markersize=2 * np.sqrt(size))[0] for size in sizes_Roc]
trails_Roc = [ax.plot([], [], '-', color='red', alpha=0.5)[0] for _ in range(N_Roc)]
points_Gas = [ax.plot([], [], 'o', color='blue', markersize=2 * np.sqrt(size))[0] for size in sizes_Gas]
trails_Gas = [ax.plot([], [], '-', color='blue', alpha=0.5)[0] for _ in range(N_Gas)]

def init():
    # Inicializar los puntos y líneas para ser vacíos al principio
    for point, trail in zip(points_Roc + points_Gas, trails_Roc + trails_Gas):
        point.set_data([], [])
        trail.set_data([], [])
    return points_Roc + trails_Roc + points_Gas + trails_Gas

def update(frame):
    # Actualizar la posición de cada planetesimal y su estela
    for i in range(N_Roc):
        points_Roc[i].set_data([r_Roc[i][0][frame]], [r_Roc[i][1][frame]])
        xdata, ydata = trails_Roc[i].get_data()
        trails_Roc[i].set_data(np.append(xdata, r_Roc[i][0][frame]), np.append(ydata, r_Roc[i][1][frame]))
        
    for i in range(N_Gas):
        points_Gas[i].set_data([r_Gas[i][0][frame]], [r_Gas[i][1][frame]])
        xdata, ydata = trails_Gas[i].get_data()
        trails_Gas[i].set_data(np.append(xdata, r_Gas[i][0][frame]), np.append(ydata, r_Gas[i][1][frame]))

    return points_Roc + trails_Roc + points_Gas + trails_Gas

# Crear y ejecutar la animación
animation = FuncAnimation(fig, update, frames=iteraciones, init_func=init, blit=True, interval=interval)

if save_to_file:
    animation.save(f"{file_out}.mp4", dpi=dpi)
else:
    plt.show()