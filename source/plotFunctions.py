import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from source import classifiers



def plotDistributions(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)

    for X in distributions:
        #reducing to 2-dimensional data
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none', zorder=2))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotDistributionByClass(instances, indexesByClass):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)

    for c, indexes in indexesByClass.items():
        X = instances[indexes]
        #reducing to 2-dimensional data
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotAccuracy(arr, steps, label):
    arr = np.array(arr)
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arr, 'k', zorder=2)
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps+1, 10))
    plt.title(label)
    plt.ylabel("Acurácia")
    plt.xlabel("Step")
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)

    for k, v in distributions.items():
        points = distributions[k]

        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()

def plotDistributionsTimesteps(instances, labels, timeSteps, markerSize=30,
                               numberOfBtaches=100, figsize =(12,3)):
    if instances.shape[1] != 2:
        print("Instances must have 2 dimensions")
        return

    fig = plt.figure(figsize=figsize )
    handles = []
    colors = ['tomato', 'steelblue', 'mediumseagreen', 'khaki', 'mediumorchid']
    markers = ['o', 'x', 's', 'v', 'd']
    numberTimeSteps = len(timeSteps)
    instancesPerBatch = int(len(instances) / numberOfBtaches)
    XMax = np.max(instances[:,0]) + 1
    XMin = np.min(instances[:,0]) - 1
    YMax = np.max(instances[:,1]) + 1
    YMin = np.min(instances[:,1]) - 1
    
    for t, n in zip(timeSteps, range(numberTimeSteps)):
        startBatch = t * instancesPerBatch
        endBatch = (t+1) * instancesPerBatch
        batch = instances[startBatch:endBatch, :]
        batchLabels = labels[startBatch:endBatch]
        ax = fig.add_subplot(1, numberTimeSteps, n+1)
        
        indexesByClass = dict()
        for c in np.unique(labels):
            indexesByClass[c] = np.where(batchLabels == c)
    
        for c, i in zip(indexesByClass.keys(), range(len(indexesByClass))):
            indexes = indexesByClass[c]
            handles.append(ax.scatter(batch[indexes, 0], batch[indexes, 1], 
                                      color=colors[i], s=markerSize,
                                      marker=markers[i], edgecolor='none',
                                      zorder=3))
            plt.xlim(XMin, XMax)
            plt.ylim(YMin, YMax)
            ax.tick_params(axis=u'both', which=u'both', labelsize=0,
                           length=0, grid_alpha=0.5)
            plt.grid(True, zorder=1)

    #ax.legend(handles, classes)
    plt.show()

def plot(X, y, coreX, coreY, t):
    classes = list(set(y))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color], zorder=2))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plotAnimation(i):
    classes = list(set(arrY[i]))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = arrX[i][np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color], zorder=2))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plot2(X, y, t, classes):
    X = classifiers.pca(X, 2)
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color], zorder=2))
        #core support points
        color+=1
        #labels
        classLabels.append('Class {}'.format(cl))

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def finalEvaluation(arrAcc, steps, label):
    print("Acurácia Média: ", np.mean(arrAcc))
    print("Desvio Padrão: ", np.std(arrAcc))
    print("Variância: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, steps, label)


def plotF1(arrF1, steps, label):
    arrF1 = np.array(arrF1)
    c = range(len(arrF1))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arrF1, 'k', zorder=2)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if steps > 10:
        plt.xticks(range(1, steps+1, 10))
    else:
        plt.xticks(range(1, steps+1))
    plt.title(label)
    plt.ylabel("F1")
    plt.xlabel("Step")
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotBoxplot(mode, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.boxplot(data, labels=labels, zorder=2)
    plt.xticks(rotation=90)

    if mode == 'acc':
        plt.title("Acurácia - Boxplot")
        #plt.xlabel('step (s)')
        plt.ylabel('Acurácia')
    elif mode == 'mcc':
        plt.title('Mathews Correlation Coefficient - Boxplot')
        plt.ylabel("Mathews Correlation Coefficient")
    elif mode == 'f1':
        plt.title('F1 - Boxplot')
        plt.ylabel("F1")

    plt.show()


def plotAccuracyCurves(listOfAccuracies, listOfMethods):
    limit = len(listOfAccuracies[0])+1

    for acc in listOfAccuracies:
        acc = np.array(acc)
        c = range(len(acc))
        ax = plt.axes()
        ax.plot(c, acc, zorder=2)

    plt.title("Curva de Acurácia")
    plt.legend(listOfMethods, bbox_to_anchor = (1.05,1))
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.xticks(range(0, limit, 10))
    plt.ylabel("Acurácia")
    plt.xlabel("Step")
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotBars(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], label=listOfMethods[l], align='center', zorder=2)

    plt.title("Tempo de Processamento total")
    plt.legend(listOfMethods, bbox_to_anchor = (1.6,1))
    plt.xlabel("Métodos")
    plt.ylabel("Tempo de execução")
    plt.xticks(range(len(listOfTimes)))
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotBars2(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], zorder=2)

    plt.title("Acurácia Média")
    plt.xlabel("Métodos")
    plt.ylabel("Acurácia")
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfTimes)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotBars3(listOfAccuracies, listOfMethods):
    
    for l in range(len(listOfAccuracies)):    
        ax = plt.axes()
        ax.bar(l, 100-listOfAccuracies[l], zorder=2)

    plt.title("Erro Médio")
    plt.xlabel("Métodos")
    plt.ylabel("Erro")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfAccuracies)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid(zorder=1, alpha=0.5)
    plt.show()


def plotBars4(baseline, listOfAccuracies, listOfMethods):
    sortedAcc =  sorted(listOfAccuracies, reverse=True)
    rank = [sortedAcc.index(x)+1 for x in listOfAccuracies]
    
    for l in range(1,len(listOfAccuracies)):    
        ax = plt.axes()
        #ax.bar(l, (listOfAccuracies[l]-baseline)/listOfAccuracies[l])
        ax.bar(l, ((listOfAccuracies[l]-baseline)/baseline)*100, zorder=2)
        print('Pos {} - Redução do Erro ({}):{}'.format(rank[l], 
              listOfMethods[l],((listOfAccuracies[l]-baseline)/baseline)*100))

    plt.title("Porcentagem de Redução do Erro")
    plt.xlabel("Métodos")
    plt.ylabel("% Erro comparado com baseline (Estático)")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(1, len(listOfAccuracies)), listOfMethods[1:])
    plt.xticks(rotation=90)
    plt.grid(zorder=1, alpha=0.5)
    plt.show()  