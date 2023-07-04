import pandas as pd
import statistics
import numpy as np
import os
import sys
import shutil
import random


########################### statistics #############################

def func_precision(stringList, answer):
    goal_count = 0
    found = 0
    for result in stringList:
        if result == str(answer):
            found = 1
        goal_count += 1

    return found/(goal_count-1)

def func_recall(stringList, answer):
    found = 0
    for result in stringList:
        if result == str(answer):
            found = 1
            break
    return found

def func_f1(total, stringList, answer):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in stringList[0:-1]:
        if result == str(answer):
            tp += 1
        else:
            fp += 1
    
    fn = 1 - tp
    
    # total is the number of all goals
    tn = total - tp - fp - fn
    return 2*tp/(2*tp + fp + fn)

def func_accuracy(total, stringList, answer):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in stringList[0:-1]:
        if result == str(answer):
            tp += 1
        else:
            fp += 1
    
    fn = 1 - tp
    
    # total is the number of all goals
    tn = total - tp - fp - fn
    return (tp + tn)/(tn + tp + fp + fn)


def func_bacc(total, stringList, answer):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in stringList[0:-1]:
        if result == str(answer):
            tp += 1
        else:
            fp += 1
    
    fn = 1 - tp
    
    # total is the number of all goals
    tn = total - tp - fp - fn

    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    bacc = (tpr + tnr)/2

    return bacc



# return a list of each statistics for every testing case
def calculate_statistics(rows):
    length = rows.shape[0]

    precision = []
    recall = []
    f1_score = []
    accuracy = []
    b_accuracy = []
        
    for index, row in rows.iterrows():
        
        answer = row["Real_Goal"]
        results = row["Results"].split("/")
        all_candidates = row["Cost"].split("/")
        
        total = len(all_candidates)-1   # the last one is /
        
        p = func_precision(results, answer)
        r = func_recall(results, answer)
        f = func_f1(total, results, answer)
        a = func_accuracy(total, results, answer)
        bacc = func_bacc(total, results, answer)
        
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
        accuracy.append(a)
        b_accuracy.append(bacc)
    
    return precision, recall, f1_score, accuracy, b_accuracy

#######################################################################






############################# file system helpers ########################
def reCreateDir(dirName):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dirName)
    if isExist:
        # delete
        shutil.rmtree(dirName)
    
    os.makedirs(dirName)
    

    
    
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


def run_system(phi, lamb, delta, theta, subject_id, n_features, n_clusters):

    ########## Run the experiments with PM-based recognizers #########
    # clean the result and feedback
    
    output_results = "f1_opt_results_%s.csv" % str(subject_id)
    # output_results = "tmp_results_%s.csv" % str(subject_id)
    os.system("rm -rf %s" % output_results)

    for test_id in range(30): #num_traces
        
        # test for all goals in this iteration
        for g in range(3):
            # goal_id, goal_id, percentage
            for percentage in [0.1,0.3,0.5,0.7,1.0]:
                # option -w -p -stw
                os.system("java -jar recognizer.jar -stw pmdata_%s/case_%s/model/ pmdata_%s/case_%s/test/sas_plan.%s %s %s %s %s %s %s %s" 
                          %(str(subject_id), str(test_id), str(subject_id), str(test_id), str(g), str(g), str(percentage), str(phi), str(lamb), str(delta), str(theta), str(output_results) )   )  


    os.system("rm -rf Feedback")

    ## function for PRIM
    # result_data = pd.read_csv(output_results)
    # p, r, f1, a, bacc = calculate_statistics(result_data)

    # return statistics.mean(p), statistics.mean(r), statistics.mean(f1), statistics.mean(a), statistics.mean(bacc)
    return 0

##############################################################################

if __name__ == '__main__':
    sub_id = [1,2,3,4,5,6,7,8,9,10]
    features_num_list = [29, 1, 2, 34, 32, 28, 22, 34, 23, 28]
    cluster_num_list = [70, 10, 150, 50, 90, 160, 80, 100, 100, 170]


    delta_list  = [1.7651824467475363,
         0.2359272193551653,
         4.7571204684180834,
         1.5762144603435813,
         0.8526070808785424,
         0.4982579949349338,
         0.2970251160550647,
         1.8398282038431564,
         0.09395370191132194,
         3.4952783267597463]
    lamb_list  = [1.3813885023807573,
         1.3265900742724064,
         2.9215105630339404,
         4.660410344490375,
         2.358430471210444,
         1.3845993410732702,
         1.3517505152762108,
         1.378007397971457,
         3.3290531138376758,
         3.9908835852408098]
    phi_list = [47, 2, 37, 4, 42, 77, 56, 72, 91, 26]
    theta_list  = [0.7824170924598155,
         0.8340306113890947,
         0.9585674041259294,
         0.6293604347180899,
         0.9123723785617636,
         0.8663436685722214,
         0.9362941960129322,
         0.7788576835572925,
         0.6184749109756382,
         0.9543071678016238]

    for i in range(10):

        subject_id = sub_id[i]
        n_features = features_num_list[i]
        n_clusters = cluster_num_list[i]

        # phi = phi_list[i]
        # lamb = delta_list[i]
        # delta = lamb_list[i]
        # theta = theta_list[i]

        phi = 50
        lamb = 1.5
        delta = 1.0
        theta = 0.9

        run_system(phi, lamb, delta, theta, subject_id, n_features, n_clusters)

