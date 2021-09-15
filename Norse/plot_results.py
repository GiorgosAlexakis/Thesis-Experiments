import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import glob
import os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sn
parameters_to_check=[]
import sys
parameter=sys.argv[1]
parameters_to_check.append(parameter)
classes = [ 'Hand Clapping'  ,
            'Right Hand Wave',
            'Left Hand Wave' ,
            'Right Arm CW'   ,
            'Right Arm CCW'  ,
            'Left Arm CW'    ,
            'Left Arm CCW'   ,
            'Arm Roll'       ,
            'Air Drums'      ,
            'Air Guitar'     ,
            'Other']
def flatten(t):
    return [item for sublist in t for item in sublist]
for i in parameters_to_check:
    path = 'output_data/'+i	
    file_test_acc = list(Path(path).cwd().glob("**/test_acc.npy"))[0]
    file_test_loss = list(Path(path).cwd().glob("**/test_loss.npy"))[0]
    file_total_loss = list(Path(path).cwd().glob("**/total_loss.npy"))[0]
    file_pred = list(Path(path).cwd().glob("**/pred.npy"))[0]
    file_true = list(Path(path).cwd().glob("**/true.npy"))[0]
    ######CORRECT PERC######
    data_array = np.load(file_test_acc)
    df = pd.DataFrame(data_array)

    #print(df)
    epochs=len(df)
    epochs_indices=[i for i in range(0,epochs)]
    layers=len(df.columns)
    layers_indices=['Layer '+str(i) for i in range(0,layers)]
    df.columns=layers_indices

    plot = df.plot(xticks=epochs_indices,title='Training Results (Accuracy)')
    n = 10 
    [l.set_visible(False) for (i,l) in enumerate(plot.xaxis.get_ticklabels()) if i % n != 0]
    plt.grid()
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Correct Predictions%")
    fig = plot.get_figure()
    filepath='./plots/correct_perc/'+i+'/acc_'+i+'.png'
    fig.savefig(filepath)
    
    ######test LOSS######
    data_array = np.load(file_test_loss)
    df = pd.DataFrame(data_array)
    #print(df)
    epochs=len(df)
    epochs_indices=[i for i in range(0,epochs)]
    layers=len(df.columns)
    layers_indices=['Layer '+str(i) for i in range(0,layers)]
    df.columns=layers_indices
    plot = df.plot(xticks=epochs_indices,title='Training Results (Test Loss)')
 
    [l.set_visible(False) for (i,l) in enumerate(plot.xaxis.get_ticklabels()) if i % n != 0]
    plt.grid()
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Test Loss")
    fig = plot.get_figure()
    filepath='./plots/test_loss/'+i+'/test_loss_'+i+'.png'
    fig.savefig(filepath)
    ######total LOSS######
    data_array = np.load(file_total_loss)
    df = pd.DataFrame(data_array)
    #print(df)
    epochs=len(df)
    epochs_indices=[i for i in range(0,epochs)]
    layers=len(df.columns)
    layers_indices=['Layer '+str(i) for i in range(0,layers)]
    df.columns=layers_indices
    plot = df.plot(xticks=epochs_indices,title='Training Results (Total Loss)')
   
    [l.set_visible(False) for (i,l) in enumerate(plot.xaxis.get_ticklabels()) if i % n != 0]
    plt.grid()
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Total Loss")
    fig = plot.get_figure()
    filepath='./plots/total_loss/'+i+'/total_loss_'+i+'.png'
    fig.savefig(filepath)
    ######CONFUSION MATRIX######
    
    true_labels = flatten(np.load(file_true,allow_pickle=True).flatten())
    pred_labels = flatten(flatten(np.load(file_pred,allow_pickle=True)))

    cf_matrix=confusion_matrix(true_labels, pred_labels)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
  
    df_cm_count = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    df_cm = pd.DataFrame(cf_matrix/cf_matrix.sum(1), index = [i for i in classes], columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))
    plot = sn.heatmap(df_cm, annot=True)
    plot.set_xlabel("Predicted Label")
    plot.set_ylabel("True Label")
    plt.xticks(rotation=15)
    title = "Confusion Matrix (Percentage)" 
    plt.title(title)
    fig = plot.get_figure()
    filepath='./plots/confusion_matrices/'+i+'/perc_'+i+'.png'
    fig.savefig(filepath)
    
    plt.figure(figsize = (12,7))
    plot1 = sn.heatmap(df_cm_count, annot=True)
    plot1.set_xlabel("Predicted Label")
    plot1.set_ylabel("True Label")
    title1 = "Confusion Matrix (Count)"
    plt.title(title1)
    plt.xticks(rotation=15)
    fig = plot1.get_figure()
    filepath='./plots/confusion_matrices/'+i+'/count_'+i+'.png'
    fig.savefig(filepath)
    
    ######Horizontal Bar######
    df = pd.DataFrame({'Classes': classes,'Class Accuracy(%)': class_accuracy})
    #print(class_accuracy)
    plt.figure(figsize = (12,7))
    
    class_acc_plot = df.plot.barh(x='Classes', y='Class Accuracy(%)')
    
    fig = class_acc_plot.get_figure()
    filepath='./plots/bar/'+i+'/barh_acc_'+i+'.png'
    fig.savefig(filepath ,bbox_inches="tight")
    plt.close('all')
        
 
