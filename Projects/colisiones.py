#Comienzo el programa incorporando las librerias necesarias para su correcto funcionamiento.
import numpy as np
import random
import numba
from numba import set_num_threads, jit

#Defino el número de nucleos que usará en joel
set_num_threads(4)

#Comenzaré el programa definiendo las condiciones iniciales. Empezaré definiendo el número de iteraciones, el paso, y el vector de tiempo.
iteraciones=10 #Número de iteraciones
h=0.01 #Paso de cada iteración
tiempo=np.zeros(iteraciones) #Matriz de tiempos 
tiempo_New=np.zeros(iteraciones) #Matriz de tiempos tras colisionar

#Ahora defino la masa de la estrella. Piden que sea una estrella masiva. He consultado que se considera estrella masiva y he concluido que debe ser 
#de 7 a 10 veces mayor que la masa del sol. Para seguir una distribución similar a la del sistema solar, supondremos que esta masa conforma el 99.86% 
#de la masa del sistema.
Ms=2.167*10**31 #Kg

#Ahora debo distinguir entre planetesimales rocosos, que conforman el 10% del total, y gaseosos que conforman el 90%.
#Defino el tamaño que tendrá la matriz de planetesimales rocosos y gaseosos
planetesimales=10 #Número de planetesimales

N_Roc=int((10*planetesimales)/100) #Número de planetesimales rocosos
N_Gas=int((90*planetesimales)/100) #Número de planetesimales gaseosos

#Defino una función para guardar en un archivo de texto la cantidad de planetesimales que  hay antes y despues de las colisiones.
def save_planet_counts(N_Roc, N_Gas, filename):
    with open(filename, 'w') as file:
        file.write(f"{N_Roc}\n")
        file.write(f"{N_Gas}\n")
        
#Guardo el número de planetesimales inicial
save_planet_counts(N_Roc, N_Gas, 'Numero_Planetas_Antiguos.txt')

#Creo la matriz de plantesimales rocosos y gaseosos
r_Roc=np.zeros((N_Roc, 2, iteraciones))
r_Gas=np.zeros((N_Gas, 2, iteraciones))

#El 0.14% de la masa restante pertenecerá a los planetesimales, que calculándola obtenemos que es 3.038*10^28. Por tanto la masa de cada
#planetesimal será 3.038*10^28/planetesimales
Mo=(3.038*10**28)/planetesimales #Kg

Masa_Roc=np.zeros(N_Roc) #Matriz de masas de los planetesimales rocosos
Masa_Gas=np.zeros(N_Gas) #Matriz de masas de los planetesimales gaseosos

for i in range(N_Roc):
    Masa_Roc[i]=Mo
for i in range(N_Gas):
    Masa_Gas[i]=Mo

#Además todos los planetesimales deben tener inicialmente el mismo radio. Esto es un poco incoherente, ya que si los rocosos y los gaseosos tienen la
#misma masa, el radio de los gaseosos debería ser mucho mayor. La media de las densidades de los planetas rocosos del sistema solar es 5030 kg/m^3, 
#mientras que la de los gaseosos es 1230 kg/m^3, por tanto cogeré el promedio de ambas, 3130 kg/m^3. Con el valor de la densidad, y suponiendo que los
#planetesimales son esféricos calcularé su radio.
Radio=(1/1000)*((3*Mo)/(4*np.pi*3130))**(1/3) #Km

Radio_Roc=np.full(N_Roc, Radio)
Radio_Gas=np.full(N_Gas, Radio)

#Para que el programa funcione correctamente debo generar los planetesimales aleatoriamente de modo que los rocosos se generen cerca del sol y los 
#gaseosos alejados. Para ello definiré un límite, que podría ser un cinturón de asteroides, dentro del cual se generaran los planetas rocosos y fuera
#los gaseosos.
Cinturon=4.03*10**7 #km

#Ahora defino la posición incial de los planetesimales. Definiré aleatoriamente un radio y ángulo, y posteriormente los convertiré a coordenadas
#cartesianas.
rho_Roc=np.zeros(N_Roc) #Defino rho para los rocosos
rho_Gas=np.zeros(N_Gas) #Defino rho para los gaseosos

theta_Roc=np.zeros(N_Roc) #Defino theta para los rocosos
theta_Gas=np.zeros(N_Gas) #Defino theta para los gaseosos

for i in range(N_Roc):
    rho_Roc[i]=random.uniform(0.5*10**7, 3.5*10**7)  #Le doy valores aleatorios
for i in range(N_Gas):
    rho_Gas[i]=random.uniform(6*10**7, 5*10**9)  #Le doy valores aleatorios
for i in range(N_Roc):
    theta_Roc[i]=np.random.random()*2*np.pi  #Le doy valores aleatorios
for i in range(N_Gas):
    theta_Gas[i]=np.random.random()*2*np.pi  #Le doy valores aleatorios

#Paso los datos a coordenadas cartesianas y defino el número de iteraciones
for i in range(N_Roc):
    r_Roc[i][0][0]=rho_Roc[i]*np.cos(theta_Roc[i])
    r_Roc[i][1][0]=rho_Roc[i]*np.sin(theta_Roc[i])

for i in range(N_Gas):
    r_Gas[i][0][0]=rho_Gas[i]*np.cos(theta_Gas[i])
    r_Gas[i][1][0]=rho_Gas[i]*np.sin(theta_Gas[i])
    
#Defino las variables de reescalamiento
c=1.496*10**8  #Km
Ms=1.99*10**30  #Kg
G=6.67384*10**(-20) #Km^3 / Kg s^2

#Reescalo masas
Masa_Roc=Masa_Roc/Ms
Masa_Gas=Masa_Gas/Ms
Mo=Mo/Ms
Ms=Ms/Ms


#Reescalo distancias
Cinturon=Cinturon/c
rho_Roc=rho_Roc/c
rho_Gas=rho_Gas/c
r_Roc=r_Roc/c
r_Gas=r_Gas/c  
Radio_Roc=Radio_Roc/c
Radio_Gas=Radio_Gas/c

#Defino la excentricidad de manera aleatoria para cada decimal, de modo que entren en un rango de [0.001,0.4]. Establezco este rango, ya que cuanto 
#mayor sea la excentricidad, mas pronunciada será la elipse de la órbita del planetesimal, y por tanto será menos probable que este colisione con otro.
ex_Roc=np.zeros(N_Roc)
ex_Gas=np.zeros(N_Gas)
for i in range(N_Roc):
    ex_Roc[i]=random.random()*0.999
for i in range(N_Gas):
    ex_Gas[i]=random.random()*0.999

#Ahora necesito obtener los vectores de la velocidad inicial y la aceleración inicial para poder comenzar con el algoritmo de Verlet. Para calcular el 
#vector velocidad inicial empezaré con su módulo. Para ello usaré la fórmula de la velocidad orbital de una elipse. En esta fórmula aparece el semieje
#mayor. Para calcularlo usaré la excentricidad y supondré que el semieje menor b es el radio inicial ro. 
SemiejeA_Roc=np.zeros(N_Roc)
SemiejeA_Gas=np.zeros(N_Gas)
for i in range(N_Roc):
    SemiejeA_Roc[i]=np.sqrt((rho_Roc[i]**2)/(1-ex_Roc[i]**2))
for i in range(N_Gas):
    SemiejeA_Gas[i]=np.sqrt((rho_Gas[i]**2)/(1-ex_Gas[i]**2))

#Con el valor del semieje mayor puedo proceder a calcular el módulo de la velcidad orbital.
moduloV_Roc=np.zeros(N_Roc)
moduloV_Gas=np.zeros(N_Gas)
for i in range(N_Roc):
    moduloV_Roc[i]=np.sqrt(2*Ms*((1/rho_Roc[i])-(1/(2*SemiejeA_Roc[i]))))
for i in range(N_Gas):
    moduloV_Gas[i]=np.sqrt(2*Ms*((1/rho_Gas[i])-(1/(2*SemiejeA_Gas[i]))))

#Una vez calculado el módulo de la velocidad inical de cada planetesimal aprovecharé la ecuación \vec{v}=|v|e^{i phi}, de modo que calculando 
#el ángulo phi obtendré la velocidad en cada eje. En mi informe se puede consultar un esquema de como se ha calculado phi en cada punto.
phi_Roc=np.zeros(N_Roc)
phi_Gas=np.zeros(N_Gas)

v_Roc=np.zeros((N_Roc, 2, iteraciones)) #Matriz de velocidades de los planetesimales rocosos
v_Gas=np.zeros((N_Gas, 2, iteraciones)) #Matriz de velocidades de los planetesimales gaseosos

for i in range(N_Roc):
    phi_Roc[i]=2*np.pi-theta_Roc[i] #Ángulo phi
    
    v_Roc[i][0][0]=moduloV_Roc[i]*np.sin(phi_Roc[i])
    v_Roc[i][1][0]=moduloV_Roc[i]*np.cos(phi_Roc[i])

for i in range(N_Gas):
    phi_Gas[i]=2*np.pi-theta_Gas[i] #Ángulo phi
    
    v_Gas[i][0][0]=moduloV_Gas[i]*np.sin(phi_Gas[i])
    v_Gas[i][1][0]=moduloV_Gas[i]*np.cos(phi_Gas[i])
    
#Ahora paso a definir la aceleración inicial con la fórmula proporcionada en el guión. Definiré la aceleración en una fórmula ya que la utilizaré en
#más ocasiones.
@jit(nopython=True)
def aceleracion(Masa, Distancia1, Distancia2):
        return-(Masa*Distancia1)/(Distancia1**2+Distancia2**2)**(3/2)
        
#Aquí calculo la aceleración inicial.
a_Roc=np.zeros((N_Roc, 2, iteraciones))   
a_Gas=np.zeros((N_Gas, 2, iteraciones)) 
for i in range(N_Roc):
        a_Roc[i][0][0]=aceleracion(Ms, r_Roc[i][0][0], r_Roc[i][1][0])
        a_Roc[i][1][0]=aceleracion(Ms, r_Roc[i][1][0], r_Roc[i][0][0])    
for i in range(N_Gas):
        a_Gas[i][0][0]=aceleracion(Ms, r_Gas[i][0][0], r_Gas[i][1][0])
        a_Gas[i][1][0]=aceleracion(Ms, r_Gas[i][1][0], r_Gas[i][0][0]) 
    
#A continuación definiré una serie de funciones que utilizaré para colisionar los planetesimales.
@jit(nopython=True)       
def detectar_planet(r, Radio, N):
    #Defino dos listas donde guardo los índices de los planetas que chocan.
    fusionar = set()
    eliminar = set() 
    for i in range(N):
        for j in range(i+1, N): #Lo defino así para que no examine 2 o más veces el  mismo par de planetas, y que no se evalúe un planeta consigo mismo
                distancia = np.linalg.norm(r[i, :] - r[j, :])
                if distancia < (Radio[i] + Radio[j]):
                    fusionar.add(i)
                    eliminar.add(j) 
    return list(fusionar), list(eliminar) # Convertimos el conjunto a lista antes de retornarlo

@jit(nopython=True)
def fusionar_planet(indice1, indice2, v, Radio, Masa, ECal):
    #Esto ocurre al ser choques inelasticos
    for i, j in zip(indice1, indice2):
        #Energía cinética inicial
        Eki=0
        Ekf=0
        Eki = 0.5 * Masa[i] * np.sum(v[i, :]**2) + 0.5 * Masa[j] * np.sum(v[j, :]**2)

        #Masa total
        total_mass = Masa[i] + Masa[j]
        
        #Velocidad final
        v[i, :] = (v[i, :] * Masa[i] + v[j, :] * Masa[j]) / total_mass

        #Masa final
        Masa[i] = total_mass

        #Radio final
        Radio[i] = np.cbrt((Radio[i]**3 * Masa[i] + Radio[j]**3 * Masa[j]) / total_mass)

        #Energía cinética final
        Ekf = 0.5 * Masa[i] * np.sum(v[i, :]**2) 
        
        #Calcular energía perdida
        ECal[i]=Eki-Ekf+ECal[j]
        
def eliminar_planet(indices_to_remove, r, v, a, Radio, Masa, ECal):
    # Ordenamos los índices a eliminar de mayor a menor para evitar desplazar los índices no procesados aún
    indices_to_remove.sort(reverse=True)
    for index in indices_to_remove:
        # Elimino los planetesimales j que ya han chocado.
        r = np.delete(r, index, axis=0)  # r es [indice_planeta, componente_xy, tiempo]
        v = np.delete(v, index, axis=0)  # v es [indice_planeta, componente_xy, tiempo]
        a = np.delete(a, index, axis=0)  # a es [indice_planeta, componente_xy, tiempo]
        Radio = np.delete(Radio, index)  # Radio es [indice_planeta]
        Masa = np.delete(Masa, index)    # Masa es [indice_planeta]
        ECal = np.delete(ECal, index)    # Energía calorífica es [indice_planeta]
        
    return r, v, a, Radio, Masa, ECal  # Retornar las matrices actualizadas


#Empezamos el algoritmo de verlet
#Defino unas matrices para las energías caloríficas tras chocar de cada planeta.
ECal_Roc=np.zeros(N_Roc)
ECal_Gas=np.zeros(N_Gas)

for k in range(iteraciones-1):
    #Evaluar r(t+h) y w(t)
    for i in range(N_Roc):
        r_Roc[i][0][k+1]=r_Roc[i][0][k]+h*v_Roc[i][0][k]+(a_Roc[i][0][k]*h**2)/2
        r_Roc[i][1][k+1]=r_Roc[i][1][k]+h*v_Roc[i][1][k]+(a_Roc[i][1][k]*h**2)/2

    for i in range(N_Gas):
        r_Gas[i][0][k+1]=r_Gas[i][0][k]+h*v_Gas[i][0][k]+(a_Gas[i][0][k]*h**2)/2
        r_Gas[i][1][k+1]=r_Gas[i][1][k]+h*v_Gas[i][1][k]+(a_Gas[i][1][k]*h**2)/2
        
    #Evaluar a(t+h) para r(t+h)
    for i in range(N_Roc):
        a_Roc[i][0][k+1]=aceleracion(Ms, r_Roc[i][0][k+1], r_Roc[i][1][k+1])
        a_Roc[i][1][k+1]=aceleracion(Ms, r_Roc[i][1][k+1], r_Roc[i][0][k+1]) 

    for i in range(N_Gas):
        a_Gas[i][0][k+1]=aceleracion(Ms, r_Gas[i][0][k+1], r_Gas[i][1][k+1])
        a_Gas[i][1][k+1]=aceleracion(Ms, r_Gas[i][1][k+1], r_Gas[i][0][k+1]) 

    #Evaluar v(t+h) 
    for i in range(N_Roc):
        v_Roc[i][0][k+1]=v_Roc[i][0][k]+(h/2)*(a_Roc[i][0][k]+a_Roc[i][0][k+1])
        v_Roc[i][1][k+1]=v_Roc[i][1][k]+(h/2)*(a_Roc[i][1][k]+a_Roc[i][1][k+1])

    for i in range(N_Gas):
        v_Gas[i][0][k+1]=v_Gas[i][0][k]+(h/2)*(a_Gas[i][0][k]+a_Gas[i][0][k+1])
        v_Gas[i][1][k+1]=v_Gas[i][1][k]+(h/2)*(a_Gas[i][1][k]+a_Gas[i][1][k+1])

    # Detectar colisiones
    indices_fusionar_Roc, indices_eliminar_Roc = detectar_planet(r_Roc[:, :, k+1], Radio_Roc[:], N_Roc)
    indices_fusionar_Gas, indices_eliminar_Gas = detectar_planet(r_Gas[:, :, k+1], Radio_Gas[:], N_Gas)

    if len(indices_fusionar_Roc)>0:
        # Fusionar 
        fusionar_planet(indices_fusionar_Roc[:], indices_eliminar_Roc[:], v_Roc[:, :, k+1], Radio_Roc[:], Masa_Roc[:], ECal_Roc[:])
    
        #Eliminar
        r_Roc, v_Roc, a_Roc, Radio_Roc, Masa_Roc, ECal_Roc\
        = eliminar_planet(indices_eliminar_Roc, r_Roc, v_Roc, a_Roc, Radio_Roc, Masa_Roc, ECal_Roc)
            
        N_Roc -= len(indices_eliminar_Roc)

    if len(indices_fusionar_Gas)>0:
        # Fusionar 
        fusionar_planet(indices_fusionar_Gas[:], indices_eliminar_Gas[:], v_Gas[:, :, k+1], Radio_Gas[:], Masa_Gas[:], ECal_Gas[:])
    
        #Eliminar
        r_Gas, v_Gas, a_Gas, Radio_Gas, Masa_Gas, ECal_Gas\
        = eliminar_planet(indices_eliminar_Gas, r_Gas, v_Gas, a_Gas, Radio_Gas, Masa_Gas, ECal_Gas)
        
        N_Gas -= len(indices_eliminar_Gas)
        
    #t=t+h
    tiempo[k+1]=tiempo[k]+h
    
# Una vez colisionados todos los planetesimales guardo el número de planetesimales resultante
save_planet_counts(N_Roc, N_Gas, 'Numero_Planetas_Nuevos.txt')

#Defino otra función para guardar todas las magnitudes de interés
def guardar_valores(Valor, filename):
    with open(filename, 'w') as file:
        for val in Valor:
            file.write(f"{val}\n")

# Llamada a la función
guardar_valores(ECal_Roc,'energy_calorifica_Roc.txt')
guardar_valores(ECal_Gas,'energy_calorifica_Gas.txt')
guardar_valores(Radio_Roc,'Radio_Roc.txt')
guardar_valores(Radio_Gas,'Radio_Gas.txt')
guardar_valores(Masa_Roc,'Masa_Roc.txt')
guardar_valores(Masa_Gas,'Masa_Gas.txt')



#Para estudiar el periodo y la excentricidad no puedo usar los datos obtenidos de r para las colisiones, ya que los planetas cambian sus orbitas 
#tras cada colisión. Por ello realizaré de nuevo un algoritmo de verlet solo con los planetas finales. Para ello copiaré las ultimas posiciones,
#velocidades y aceleraciones de los planetas resultantes.
#Defino las nuevas magnitudes
r_Roc_New=np.zeros((N_Roc, 2, iteraciones))
r_Gas_New=np.zeros((N_Gas, 2, iteraciones))

v_Roc_New=np.zeros((N_Roc, 2, iteraciones))
v_Gas_New=np.zeros((N_Gas, 2, iteraciones))

a_Roc_New=np.zeros((N_Roc, 2, iteraciones))
a_Gas_New=np.zeros((N_Gas, 2, iteraciones))

#Copio el último elemento de mis antiguas matrices en el primer elemento de las nuevas
for i in range(N_Roc):
    r_Roc_New[i,:,0]=r_Roc[i,:,iteraciones-1]
    v_Roc_New[i,:,0]=v_Roc[i,:,iteraciones-1]
    a_Roc_New[i,:,0]=a_Roc[i,:,iteraciones-1]

for i in range(N_Gas):
    r_Gas_New[i,:,0]=r_Gas[i,:,iteraciones-1]
    v_Gas_New[i,:,0]=v_Gas[i,:,iteraciones-1]
    a_Gas_New[i,:,0]=a_Gas[i,:,iteraciones-1]

#Cominezo el nuevo algoritmo de verlet
for k in range(iteraciones-1):
    #Evaluar r(t+h) y w(t)
    for i in range(N_Roc):
        r_Roc_New[i][0][k+1]=r_Roc_New[i][0][k]+h*v_Roc_New[i][0][k]+(a_Roc_New[i][0][k]*h**2)/2
        r_Roc_New[i][1][k+1]=r_Roc_New[i][1][k]+h*v_Roc_New[i][1][k]+(a_Roc_New[i][1][k]*h**2)/2

    for i in range(N_Gas):
        r_Gas_New[i][0][k+1]=r_Gas_New[i][0][k]+h*v_Gas_New[i][0][k]+(a_Gas_New[i][0][k]*h**2)/2
        r_Gas_New[i][1][k+1]=r_Gas_New[i][1][k]+h*v_Gas_New[i][1][k]+(a_Gas_New[i][1][k]*h**2)/2
        
    #Evaluar a(t+h) para r(t+h)
    for i in range(N_Roc):
        a_Roc_New[i][0][k+1]=aceleracion(Ms, r_Roc_New[i][0][k+1], r_Roc_New[i][1][k+1])
        a_Roc_New[i][1][k+1]=aceleracion(Ms, r_Roc_New[i][1][k+1], r_Roc_New[i][0][k+1]) 

    for i in range(N_Gas):
        a_Gas_New[i][0][k+1]=aceleracion(Ms, r_Gas_New[i][0][k+1], r_Gas_New[i][1][k+1])
        a_Gas_New[i][1][k+1]=aceleracion(Ms, r_Gas_New[i][1][k+1], r_Gas_New[i][0][k+1]) 

    #Evaluar v(t+h) 
    for i in range(N_Roc):
        v_Roc_New[i][0][k+1]=v_Roc_New[i][0][k]+(h/2)*(a_Roc_New[i][0][k]+a_Roc_New[i][0][k+1])
        v_Roc_New[i][1][k+1]=v_Roc_New[i][1][k]+(h/2)*(a_Roc_New[i][1][k]+a_Roc_New[i][1][k+1])

    for i in range(N_Gas):
        v_Gas_New[i][0][k+1]=v_Gas_New[i][0][k]+(h/2)*(a_Gas_New[i][0][k]+a_Gas_New[i][0][k+1])
        v_Gas_New[i][1][k+1]=v_Gas_New[i][1][k]+(h/2)*(a_Gas_New[i][1][k]+a_Gas_New[i][1][k+1])

    #t=t+h
    tiempo_New[k+1]=tiempo_New[k]+h
    
#Defino una función para guardar los valores de r.
def save_to_file(data, filename, N, iteraciones):
    with open(filename, 'w') as file:
        for i in range(N):
            for k in range(iteraciones):
                # Escribe las coordenadas x e y para la iteración k del elemento i
                file.write(f"{data[i][0][k]}, {data[i][1][k]}\n")
            file.write("\n")  # Añade una línea vacía después de cada elemento para mejor visualización


#Guardo los valores
save_to_file(r_Roc_New, 'r_Roc_New.txt', N_Roc, iteraciones)
save_to_file(r_Gas_New, 'r_Gas_New.txt', N_Gas, iteraciones)

#Ahora pasaré a calcular el periodo y la excentricidad, para ello buscaré los máximos y mínimos de cada planetesimal para posteriormente usarlos en 
#operaciones. Además obtendré el número de vueltas que realiza cada planetesimal.
# Inicializar arrays para almacenar los máximos, mínimos, y conteo de vueltas de los planetas rocosos
vueltas_Roc = np.zeros(N_Roc)
xmax_Roc = np.zeros(N_Roc)
xmin_Roc = np.zeros(N_Roc)
ymax_Roc = np.zeros(N_Roc)
ymin_Roc = np.zeros(N_Roc)
rmax_Roc = np.zeros(N_Roc)
rmin_Roc = np.full(N_Roc, 50.0)  # Valor inicial de rmin para los planetas rocosos
modulo_Roc = np.zeros((N_Roc, iteraciones))

# Calculamos el tiempo transcurrido y lo convertimos a días
t_Roc = tiempo_New[iteraciones-1] * (c**3/(G*2.167*10**31))**0.5 
t_Roc /= (3600 * 24)  # Conversión de segundos a días

for i in range(N_Roc):
    for k in range(iteraciones - 1):
        # Calcular el módulo de la posición para planetas rocosos
        modulo_Roc[i][k] = np.sqrt(r_Roc_New[i][0][k]**2 + r_Roc_New[i][1][k]**2)
        if (r_Roc_New[i][0][k] * r_Roc_New[i][0][k+1] < 0) and (r_Roc_New[i][0][k] > r_Roc_New[i][0][k+1]):
            vueltas_Roc[i] += 1
        if vueltas_Roc[i] <= 1:  # Solo actualiza máximos y mínimos si no ha completado una vuelta
            xmax_Roc[i] = max(xmax_Roc[i], r_Roc_New[i][0][k])
            xmin_Roc[i] = min(xmin_Roc[i], r_Roc_New[i][0][k])
            ymax_Roc[i] = max(ymax_Roc[i], r_Roc_New[i][1][k])
            ymin_Roc[i] = min(ymin_Roc[i], r_Roc_New[i][1][k])
            rmax_Roc[i] = max(rmax_Roc[i], modulo_Roc[i][k])
            rmin_Roc[i] = min(rmin_Roc[i], modulo_Roc[i][k])



# Inicializar arrays para almacenar los máximos, mínimos, y conteo de vueltas de los planetas gaseosos
vueltas_Gas = np.zeros(N_Gas)
xmax_Gas = np.zeros(N_Gas)
xmin_Gas = np.zeros(N_Gas)
ymax_Gas = np.zeros(N_Gas)
ymin_Gas = np.zeros(N_Gas)
rmax_Gas = np.zeros(N_Gas)
rmin_Gas = np.full(N_Gas, 50.0)  # Valor inicial de rmin para los planetas gaseosos
modulo_Gas = np.zeros((N_Gas, iteraciones))

# Calculamos el tiempo transcurrido y lo convertimos a días
t_Gas = tiempo_New[iteraciones-1] * (c**3/(G*2.167*10**31))**0.5
t_Gas /= (3600 * 24)  # Conversión de segundos a días

for i in range(N_Gas):
    for k in range(iteraciones - 1):
        # Calcular el módulo de la posición para planetas gaseosos
        modulo_Gas[i][k] = np.sqrt(r_Gas_New[i][0][k]**2 + r_Gas_New[i][1][k]**2)
        if (r_Gas_New[i][0][k] * r_Gas_New[i][0][k+1] < 0) and (r_Gas_New[i][0][k] > r_Gas_New[i][0][k+1]):
            vueltas_Gas[i] += 1
        if vueltas_Gas[i] <= 1:  # Solo actualiza máximos y mínimos si no ha completado una vuelta
            xmax_Gas[i] = max(xmax_Gas[i], r_Gas_New[i][0][k])
            xmin_Gas[i] = min(xmin_Gas[i], r_Gas_New[i][0][k])
            ymax_Gas[i] = max(ymax_Gas[i], r_Gas_New[i][1][k])
            ymin_Gas[i] = min(ymin_Gas[i], r_Gas_New[i][1][k])
            rmax_Gas[i] = max(rmax_Gas[i], modulo_Gas[i][k])
            rmin_Gas[i] = min(rmin_Gas[i], modulo_Gas[i][k])


#Ahora calculo el semieje mayor y menor para cada planetesimal
semi_a_Roc=np.zeros(N_Roc)
semi_b_Roc=np.zeros(N_Roc)

for i in range(N_Roc):
    semi_a_Roc[i]=(xmax_Roc[i]-xmin_Roc[i])/2
    semi_b_Roc[i]=(ymax_Roc[i]-ymin_Roc[i])/2

semi_a_Gas=np.zeros(N_Gas)
semi_b_Gas=np.zeros(N_Gas)

for i in range(N_Gas):
    semi_a_Gas[i]=(xmax_Gas[i]-xmin_Gas[i])/2
    semi_b_Gas[i]=(ymax_Gas[i]-ymin_Gas[i])/2


#Calculando el número de vueltas no es suficiente para calcular el periodo correctamente, ya que puede que hayamos contado n vueltas pero un al terminar 
#las iteraciones un planetesimal se encuentre a medias de completar otra vuelta. Para ello calcularé en que parte de su orbita se queda el planetesimal
#en su última iteración, calcularé un porcentaje de como de completada está su órbita y lo sumaré al número de vueltas.
theta_Roc_New = np.zeros(N_Roc)

for i in range(N_Roc):
    if r_Roc_New[i][0][iteraciones-1] > 0 and r_Roc_New[i][1][iteraciones-1] > 0:
        theta_Roc_New[i] = np.arctan((r_Roc_New[i][1][iteraciones-1]*semi_b_Roc[i]) / (semi_a_Roc[i]*r_Roc_New[i][0][iteraciones-1]))
    elif r_Roc_New[i][0][iteraciones-1] > 0 and r_Roc_New[i][1][iteraciones-1] < 0:
        theta_Roc_New[i] = np.arctan((r_Roc_New[i][1][iteraciones-1]*semi_b_Roc[i]) / (r_Roc_New[i][0][iteraciones-1]*semi_a_Roc[i])) + 2 * np.pi
    elif r_Roc_New[i][0][iteraciones-1] < 0 and r_Roc_New[i][1][iteraciones-1] > 0:
        theta_Roc_New[i] = np.arctan((r_Roc_New[i][1][iteraciones-1]*semi_b_Roc[i]) / (r_Roc_New[i][0][iteraciones-1]*semi_a_Roc[i])) + np.pi
    else:
        theta_Roc_New[i] = np.arctan((r_Roc_New[i][1][iteraciones-1]*semi_b_Roc[i]) / (r_Roc_New[i][0][iteraciones-1]*semi_a_Roc[i])) + np.pi

vueltasF_Roc = vueltas_Roc + theta_Roc_New / (2 * np.pi)

# Cálculo de los periodos de la simulación para planetas rocosos
T_simulacion_Roc = t_Roc / vueltasF_Roc



# Calcular vueltas exactas para planetas gaseosos
theta_Gas_New = np.zeros(N_Gas)

for i in range(N_Gas):
    if r_Gas_New[i][0][iteraciones-1] > 0 and r_Gas_New[i][1][iteraciones-1] > 0:
        theta_Gas_New[i] = np.arctan((r_Gas_New[i][1][iteraciones-1]*semi_b_Gas[i]) / (r_Gas_New[i][0][iteraciones-1]*semi_a_Gas[i]))
    elif r_Gas_New[i][0][iteraciones-1] > 0 and r_Gas_New[i][1][iteraciones-1] < 0:
        theta_Gas_New[i] = np.arctan((r_Gas_New[i][1][iteraciones-1]*semi_b_Gas[i]) / (r_Gas_New[i][0][iteraciones-1]*semi_a_Gas[i])) + 2 * np.pi
    elif r_Gas_New[i][0][iteraciones-1] < 0 and r_Gas_New[i][1][iteraciones-1] > 0:
        theta_Gas_New[i] = np.arctan((r_Gas_New[i][1][iteraciones-1]*semi_b_Gas[i]) / (r_Gas_New[i][0][iteraciones-1]*semi_a_Gas[i])) + np.pi
    else:
        theta_Gas_New[i] = np.arctan((r_Gas_New[i][1][iteraciones-1]*semi_b_Gas[i]) / (r_Gas_New[i][0][iteraciones-1]*semi_a_Gas[i])) + np.pi

vueltasF_Gas = vueltas_Gas + theta_Gas_New / (2 * np.pi)

# Cálculo de los periodos de la simulación para planetas gaseosos
T_simulacion_Gas = t_Gas / vueltasF_Gas

# Uso de la función para planetas rocosos y gaseosos
guardar_valores(T_simulacion_Roc,'Periodos_Roc.txt')
guardar_valores(T_simulacion_Gas,'Periodos_Gas.txt')


#Por último calculo la excentricidad.
ex_Roc_New = np.zeros(N_Roc)

for i in range(N_Roc):
    ex_Roc_New[i] = (rmax_Roc[i] - rmin_Roc[i]) / (rmax_Roc[i] + rmin_Roc[i])


# Calcular la excentricidad de cada planeta gaseoso
ex_Gas_New = np.zeros(N_Gas)

for i in range(N_Gas):
    ex_Gas_New[i] = (rmax_Gas[i] - rmin_Gas[i]) / (rmax_Gas[i] + rmin_Gas[i])

guardar_valores(ex_Roc_New,'Ex_Roc.txt')
guardar_valores(ex_Gas_New,'Ex_Gas.txt')