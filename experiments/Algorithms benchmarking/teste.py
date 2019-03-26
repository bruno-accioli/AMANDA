import sys
import os
import numpy as np
from pathlib import Path
os.chdir(Path(os.getcwd()).resolve().parents[1])
import setup
from methods import grid_selection_amanda_fixed
from sklearn.model_selection import ParameterGrid



def writeResults(datasetID, bestScore, bestParams, clfName):
    path = "results/batch/fixed/gridsearch_amanda_fixed_20PERCENT-{}.txt".format(clfName)
    file = open(path,"a") 
    string = "{}: {} using {} \n".format(datasetID, bestScore, bestParams)
    file.write(string)
    file.close() 


def main():
    is_windows = sys.platform.startswith('win')
    sep = '\\'
    
    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'data'+sep
    
    #loading datasets
    datasets = [setup.loadCDT, setup.loadCHT, setup.load2CDT, setup.load2CHT, setup.load4CR, setup.load4CRE_V1, 
                setup.load4CRE_V2, setup.load5CVT, setup.loadCSurr, setup.load4CE1CF, setup.loadUG_2C_2D, setup.loadMG_2C_2D, 
                setup.loadFG_2C_2D, setup.loadUG_2C_3D, setup.loadUG_2C_5D, setup.loadGEARS_2C_2D, setup.loadCheckerBoard, 
                setup.loadElecData, setup.loadKeystroke, setup.loadNOAADataset]
    
    arrClfName = ['LP']
    for clfName in arrClfName:
        print("**************** BEGIN of {} results ****************".format(clfName))        
        poolSize = None
        isBatchMode = True
        #testing grid search
        
        for i in range(len(datasets)):
            splitPercentage = 0.2
            batches=16
            if i==len(datasets)-2: #NOAA
                batches=8
            elif i==len(datasets)-1: #Keystroke
                batches=4
                splitPercentage = 0.5            

            finalScore = 0
            best_grid={}
            dataValues, dataLabels, description = datasets[i](path, sep)

            #Train-test split
            availableQty = int(splitPercentage*len(dataLabels))
            availableLabels = dataLabels[:availableQty] 
            availableData = dataValues[:availableQty]

            initialLabeledData = int(0.25*len(availableLabels))
            sizeOfBatch = int((len(availableLabels)-initialLabeledData)/batches)

            print("{}: {} batches of {} instances".format(description, batches, sizeOfBatch))

            tuned_params = [{"excludingPercentage" : [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
                             "sizeOfBatch":[sizeOfBatch], "batches":[batches], "poolSize":[poolSize],
                             "isBatchMode":[isBatchMode], "initialLabeledData":[initialLabeledData], "clfName":[clfName]}]
            if clfName == 'LP' or clfName == 'KNN':
                tuned_params[0].update({"K":[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})

            for g in ParameterGrid(tuned_params):
                averageAccuracy=0
                gs = grid_selection_amanda_fixed.run(**g)

                try:
                    gs.fit(availableData, availableLabels)
                    averageAccuracy = np.mean(gs.predict())
                    print(averageAccuracy, g)
                    if finalScore < averageAccuracy:
                        finalScore = averageAccuracy
                        best_grid = g
                except Exception:
                    print("An error occured in ", description, g)
                    #raise Exception

            print(finalScore)
            print(best_grid)
            print("=======================================================================================================")

            #writeResults(description, finalScore, best_grid, clfName)
        print("******** END of {} results ********".format(clfName))
    
if __name__ == "__main__":
    main()
