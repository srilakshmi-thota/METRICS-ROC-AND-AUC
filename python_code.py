from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy import trapz

def confusion_metrics(labels,predictions,threshold):
    true_positive=0;
    false_positive=0;
    true_negative=0;
    false_negative=0;
    for i in range(len(labels)):
        if labels[i]==1:
            if predictions[i]>=threshold:
                true_positive+=1;
            else:
                false_negative+=1;
        else:
            if predictions[i]>=threshold:
                false_positive+=1;
            else:
                true_negative+=1;
    tpf=true_positive/(true_positive + false_negative);
    fpf=false_positive/(false_positive + true_negative);
    return tpf,fpf;

def results(labels,predictions):
    TPF=[];
    FPF=[];
    THRESHOLD=[];
    i=0;
    #increemental step size for threshold
    dx_step=0.0002;
    while(i<=1):
        threshold=i;
        tpf,fpf=confusion_metrics(labels,predictions,threshold);
        TPF.append(tpf);
        FPF.append(fpf);
        THRESHOLD.append(threshold);
        i+=dx_step;
        
    plt.plot(FPF,TPF);
    plt.plot(THRESHOLD,THRESHOLD,'--')
    plt.xlabel("False Positive fraction (FPF)--->")
    plt.ylabel("True Positive fraction (TPF)--->")
    plt.title("ROC Curve")
    plt.show()
    area = trapz(TPF, dx=dx_step)
    print("AUC:Area under the ROC curve is", area)
    return;
        
actual=loadmat('actual.mat');
predicted=loadmat('predicted.mat');

labels=actual['target'][:,0];
predictions=predicted['neuralOut'][:,0]; 
    
results(labels,predictions);   
    
            
        
        

