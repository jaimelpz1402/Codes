import math
import numpy as np
import random

# Definimos la matriz de spines
N = 16  # Tamaño de la matriz
S = 2*np.random.randint(2, size=(N,N))-1  # Rellenamos la matriz con 1 y -1 de forma aleatoria.
l = 10**4  # Definimos el número de iteraciones
T = 0.5  # Temperatura

#Para hacerlo en tu PC
with open("/home/cphys-jaime.lopezgonzalez/COMPU2024/datos.txt", "w") as f:
    #Empezamos el algoritmo
    for k in range(l):
        for i in range(N):
            for j in range(N):
                m = np.random.randint(0, N)
                n = np.random.randint(0, N)
                p=S[m,n]

                AE = 2*S[m,n]*(S[(m+1)%N,n] + S[m,(n+1)%N] + S[(m-1)%N,n] + S[m,(n-1)%N])
        
                if AE < 0:
                    p *= -1
                elif random.random() < np.exp(-AE/T):
                    p *= -1
                S[m,n] = p
                
        np.savetxt(f, S, fmt='%d', delimiter=',')
        f.write('\n')

   