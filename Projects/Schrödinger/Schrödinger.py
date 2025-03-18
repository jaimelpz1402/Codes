import numpy as np
import random
import cmath
import math

#***********INPUT****************
#hay que elegir el que se quiera
N=500
t_s=1000
#tiene que cumplirse que n esté entre 1 y N/4
n_c=N/4
h=0.1
s=0.1

#***********SEGUNDO CUADRO*********

k0b=(2*math.pi*n_c)/N

s=1/(4*k0b**2)


#la j va de 0 a N, es la coordenada espacial escalada

#el número de columnas es la temporal, ponemos 100

#el numero imaginario i vamos a usar la j
i=1j

phi = np.zeros((N+1, t_s),dtype=np.complex128)

#se rellena y se pone la primera condición en la primera columna t=0
#es un 8 o 2

j=0
for j in range(N+1):
    phi[j, 0] = cmath.exp((k0b*j)*i)*cmath.exp((-8*((4*j)-N)**2)/N**2)

    


def V_ (j):
    #usando ejemplo de diapositivas, valor de lambda
    l=1
    N=20
    n_c=N/4
    k0b=(2*math.pi*n_c)/N
    
    if ((((2*N)/5))<j<((3*N)/5)):
        V=l*(k0b**2)
    else:
        V=0
    return V


#CALCULO DE ALPHA  

alpha=np.zeros(N,dtype=np.complex128)

alpha[N-1]=0

j=N-1
A_menos=1
A_mas=1
i=1j

for j in range (N-1, -1, -1):
    A0=-2+((2*i)/s)-V_(j)
    alpha[j-1]=(-A_menos)/(A0+A_mas*alpha[j])


#CALCULO DE BETA

X = np.zeros((N+1, t_s),dtype=np.complex128)
       
beta = np.zeros((N, t_s),dtype=np.complex128)
n=0
i=1j
for n in range(t_s-1):
    beta[N-1,n]=0
    j=N-1
    A_menos=1
    A_mas=1
    
    f=0
    c=0
    

    for j in range (N-1, -1, -1):
        b_jn=(4*i*phi[j,n])/s 
        A0=-2+((2*i)/s)-V_(j)
        beta[j-1,n]=(b_jn-(A_mas*beta[j,n]))/(A0+(A_mas*alpha[j]))
    
    X[0,0]=0
    j=0
    
    for j in range(N-1):
        X[j+1,n]=(alpha[j])*X[j,n]+beta[j,n]
    
    j=0
    
    for j in range(N+1):
        phi[j,n+1]=X[j,n]-phi[j,n]



n=0
j=0
norma=np.zeros(t_s)

#calculo
for n in range(t_s):
    suma=0
    for j in range(N):
        suma=suma+((((abs(phi[j,n]))**2+(abs(phi[j+1,n]))**2)/2)*h)
    norma[n]=suma
        
print(norma)     
        



    
