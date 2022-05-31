##############################################################################
# OTM_VIGA.py                                                                #
# Exemplo introdutório de otimização estrutural de uma viga hiperestática:   #
# Dada uma viga com o bordo esquerdo engastado, onde posicionar os apoios    #
# A e B, de forma a minimizar o momento fletor de projeto?                   #
#                                                                            #
# Desenvolvimento: Jordlly Silva                                             #
# Criado em: 27/05/2022   Modificado em: 27/05/2022                          #
##############################################################################

import numpy as np                   # Módulo Numpy
import matplotlib.pyplot as plt      # Função Pyplot
from scipy.optimize import minimize  # Módulo Scipy

'FUNÇÃO OTIMIZAÇÃO'

### Definição da função do momento de projeto:
def f(x):
    
    # Variáveis:
    A = x[0]
    B = x[1]
    
    # Carga distribuída:
    q = 1                                     
    
    # Comprimento da viga:
    L = 10                                    
    
    # Momento no Apoio 3:
    X3 = - q*(L-B)**2/2                       
    
    # Parâmetro auxiliar 1:
    d1 = - q*A**3/4                           
    
    # Parâmetro auxiliar 2:
    d2 = - q*A**3/4 - q*(B-A)**3/4 - X3*(B-A) 
    
    # Momento no apoio 1:
    X1 = -2*B/(A**2-4*A*B)*d1 +  1/(A-4*B)*d2 
    
    # Momento no apoio 2:
    X2 =    1/(A-4*B)*d1      + -2/(A-4*B)*d2 
    
    # Cortante a esquerda 1:
    V1 = q*A/2 - X1/A + X2/A                  
    
    # Cortante a esquerda 2:
    if(A==B): 
        V2 = 0 
    else: 
        V2 = q*(B-A)/2 - X2/(B-A) + X3/(B-A)  
        
    # Momento fleto no vão 1:
    if (V1>0) and (V1/q<A):
        M1 = X1 + V1**2/2/q
    else:
        M1 = 0
        
    # Momento fleto no vão 2:
    if (V2>0) and (V2/q<(B-A)):
        M2 = X2 + V2**2/2/q
    else:
        M2 = 0
    
    # Momento de projeto:
    Md = max(abs(X1),abs(X2),abs(X3),abs(M1),abs(M2))
    
    return Md


'OTIMIZAÇÃO'

### Restrições no contorno:
bnds = []
b = (0, 10)
for i in range(0, 2):
    bnds.append(b)
bnds = tuple(bnds)
x0 = np.array([[3],[8]])

### Otimização:
solution = minimize(f, x0, method='SLSQP', bounds=bnds)
x_otimo = solution.x


'PLOTAGEM'

### Inicialização da matriz com as soluções:
Z = np.zeros((100+1, 100+1)) 

### Montagem da matriz com as soluções:
for i in range(1, 100+1):
    for j in range(1, 100+1):
        x1 = i/10
        y1 = j/10
        x = np.array([[x1],[y1]])
        if y1 >= x1:
            Z[i][j] = f(x) 
        else:
            Z[i][j] = Z[j][i] # usar o simétrico.

### Criação da malha de pontos:
x = np.linspace(0, 10, 101)
y = np.linspace(0, 10, 101)
X, Y = np.meshgrid(x, y)

### Ponto específico a ser plotado:
d1, d2 = 3, 8
D1, D2 = x_otimo[0], x_otimo[1]
  
### Plotagem:
plt.subplots(figsize=(8,6.5))
plt.contourf(X, Y, Z, 50, cmap=plt.cm.twilight_shifted)
plt.plot(d1,d2,marker='x', color='w')
plt.plot(D1,D2,marker='x', color='w')
plt.colorbar()
plt.grid(True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("POSIÇÃO DO PRIMEIRO PILAR (A)", fontsize=13)
plt.ylabel("POSIÇÃO DO SEGUNDO PILAR (B)", fontsize=13)   
#plt.axis([6, 9, 1, 5])
