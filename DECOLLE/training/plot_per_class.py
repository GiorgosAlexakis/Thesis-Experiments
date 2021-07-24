import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import glob
import os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix
def Average(lst):
    return sum(lst) / len(lst)
    
true_labels = np.column_stack(np.load('true.npy'))
pred_labels = np.column_stack(np.load('pred.npy'))
print(true_labels.shape)
print(pred_labels.shape)
cf_matrix=confusion_matrix(true_labels[0], pred_labels[3][:])
print(cf_matrix,'\n')
class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
print(class_accuracy)
print(np.mean(class_accuracy))
