# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:59:04 2019

@author: bruno.silva
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# alpha < 0 OR alpha > 1
alpha = [-20, -10, -2, -1, -0.01, 1.01, 2, 10, 20]
for a in alpha:
    b = math.pow((a / (a - 1)), a)
    psi0 = 1 / b
    valMax = -math.log(psi0, b)
    print('Alpha: %.2f | b: %.2f | Val max: %.f | psi: %.2f' %(a, b ,valMax, psi0))
      

ro = np.linspace(0, 1, 100)
alpha = [-10, -2, -0.01, -0.5, -0.001, 1.001, 1.01, 2, 10]

for a in alpha:
    bbd = np.log(1 - np.subtract(1,ro)/a) / np.log(1 - 1/a)
    plt.plot(ro, bbd, label=('\u03B2 = {:.3f}').format(a))

# a -> +-infinito, bbd = HelingerDistance^2

hellingerDistance = np.sqrt(np.subtract(1,ro))
plt.plot(ro, hellingerDistance, label='DH')

#hellingerDistance2 = np.subtract(1,ro)
#plt.plot(ro, hellingerDistance2, label='HD2')
    
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title("Comparação de valores de \u03B2", fontsize=13)
plt.xlabel("\u03C1", fontsize=12)
plt.ylabel("Distância", fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor = (1.32,1), fontsize=11)
plt.grid()    
plt.show()    
