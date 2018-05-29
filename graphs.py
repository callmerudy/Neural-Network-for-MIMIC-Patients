#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:27:16 2018

@author: nehansh
"""

import matplotlib.pyplot as plt
loss = [0.4033, 0.1786,0.1284, 0.0842, 0.0746, 0.0592, 0.0434, 0.0337, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288 ]
val_loss = [0.1377, 0.1317, 0.2362, 0.1660, 0.2359, 0.2165, 0.2760, 0.2940, 0.2766, 0.2636, 0.3122, 0.2672, 0.2820, 0.2820, 0.2820, 0.2820, 0.2820, 0.2820, 0.2820, 0.2820 ]
epochs = range(1, 21)
print(epochs)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Traning and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
    
plt.clf()
acc = [0.8563, 0.9507, 0.9739, 0.9826, 0.9904, 0.9937, 0.9952, 0.9973, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982, 0.9982  ]
val_acc = [0.9473, 0.9575, 0.9541, 0.9643, 0.9609, 0.9711, 0.9660, 0.9660, 0.9694, 0.9694, 0.9677, 0.9694, 0.9677, 0.9677, 0.9677, 0.9677, 0.9677, 0.9677, 0.9677, 0.9677 ]
    
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Traning and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()