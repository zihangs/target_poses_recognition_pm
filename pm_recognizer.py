import pandas as pd
import statistics
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

import os
import sys
import shutil
import random

########################### functions #############################
def select_features(df, labels, features, n_features):
    ## create empty lists as placeholders
    grouped_features = []
    for i in range(n_features):
        new = []
        grouped_features.append(new)

    for i in range(len(features)):
        grouped_features[labels[i]].append(features[i])

    selected_features = []
    for fs in grouped_features:
        matrix = df[fs].corr().abs()
        max_f_id = matrix.sum(axis=1).argmax()
        selected_features.append(fs[max_f_id])
        
    return selected_features





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
    accuracy = []
    b_accuracy = []
        
    for index, row in rows.iterrows():
        
        answer = row["Real_Goal"]
        results = row["Results"].split("/")
        all_candidates = row["Cost"].split("/")
        
        total = len(all_candidates)-1   # the last one is /
        
        p = func_precision(results, answer)
        r = func_recall(results, answer)
        a = func_accuracy(total, results, answer)
        bacc = func_bacc(total, results, answer)
        
        precision.append(p)
        recall.append(r)
        accuracy.append(a)
        b_accuracy.append(bacc)
    
    return precision, recall, accuracy, b_accuracy


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




# sort the df to traces for every subject and goals
def extract_traces(dfn):
    traces = []

    Subject = 0
    Loc = 0
    Iteration = 0
    tup = (Subject, Loc, Iteration)

    for index, row in dfn.iterrows():
        curr_Subject = row["Subject"]
        curr_Loc = row["Loc"]
        curr_Iteration = row["Iteration"]
        curr_tup = (curr_Subject, curr_Loc, curr_Iteration)

        if curr_tup != tup:
            #print("new trace")
            tup = curr_tup

            rslt_df = dfn[(dfn['Subject'] == curr_Subject) 
                      & (dfn['Loc'] == curr_Loc) 
                      & (dfn['Iteration'] == curr_Iteration)]

            rslt_df.reset_index(drop=True, inplace=True)
            traces.append(rslt_df)
            
    return traces


# collect labeled traces per goal
def convert_labels_kmeans(traces, goals, subject_id):
    # generate classifiers here
    
    subtraces = []
    for goal in goals:
        subtraces_goalX = []
        for t in traces:
            if t["Subject"][0] == subject_id and t["Loc"][0] == goal:
                converted_trace = []
                
                for index,e in t.iterrows():
                    converted_trace.append( e["class"] )
                                        
                subtraces_goalX.append(converted_trace)
        subtraces.append(subtraces_goalX)
            
    return subtraces



############################# file system helpers ########################
def reCreateDir(dirName):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dirName)
    if isExist:
        # delete
        shutil.rmtree(dirName)
    
    os.makedirs(dirName)
    
    
# write sas_plan
def write_plan(actions, file):
    string = ""
    for a in actions:
        string += "%s\n" % (str(a))
    string += "; cost %s (unit cost)" % (str(len(actions)))
    
    file1 = open(file, "w")
    file1.write(string)
    file1.close()
    return 0



############################ manual classifier #############################
class Classifier:
    def __init__ (self, feature, df, cls):
        self.feature = feature
        self.max_value = df[feature].max()
        self.min_value = df[feature].min()
        self.interval = (self.max_value - self.min_value)/cls
        self.lenght = len(df)
        self.subsetLen = int(self.lenght/cls)
        self.sorting_points = self.sortValues(df[feature])
        
    def sortValues(self, values):
        sorted_values = list(values.sort_values())
        sorting_points = []
        for i in range(0, self.lenght, self.subsetLen):
            sorting_points.append(i)
        return sorting_points
        
        
    def convert(self, num):
        return int( (num - self.min_value)/self.interval )
    
    def convertSorted(self, num):
        i = 0
        for p in self.sorting_points:
            if num < p:
                return i
            i += 1
        return i
    
    
    
def map2labels(df, features, n_labels):
    converted_trace = []
    for f in features:
        clf = Classifier(f,df,n_labels)
        col = []
        for index,e in df.iterrows():
            col.append(clf.convert(e[f]))
#             col.append(clf.convertSorted(e[f]))
        converted_trace.append( col )

    # transform matrix
    converted_trace = np.array(converted_trace).transpose()
    
    output = []
    for i in range(len(converted_trace)):
        output.append(str(tuple(converted_trace[i])))
    
    return np.array(output)


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


def run_system(phi, lamb, delta, theta, subject_id, n_features, n_clusters):
    # parameters:
    input_data = "final_input_data.csv"
    index_headers = 4

    # n_features = 32

    # "kmeans", "manual"
    classifier_option = "kmeans"

    # for kmeans
    n_init = 20        # the number of iterations in K-Means
    # n_clusters = 60

    # subject_id = 1


    # dependent
    output_results = "f1_def_results_%s.csv" % str(subject_id)


    ##############################################################
    df = pd.read_csv(input_data)
    all_features = df.columns.values.tolist()[index_headers::]
    df_context = df[all_features]

    corr_matrix = df_context.corr().abs()
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_features, affinity='euclidean', linkage='ward').fit(corr_matrix)
    labels = hierarchical_cluster.labels_

    selected_features = select_features(df, labels, all_features, n_features)

    ##############################################################
    df_a_subject = df.loc[df['Subject'] == subject_id].reset_index(drop=True)
    df_a_subject_index = df_a_subject[["Subject", "Loc", "Iteration"]]

    reduced = df_a_subject[selected_features]

    # classifier_option
    if classifier_option == "kmeans":
        ## K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init).fit(reduced)
        df_classes = pd.DataFrame(kmeans.labels_, columns = ['class'])

    if classifier_option == "manual":
        ### manual classification
        converted_trace = map2labels(reduced, selected_features, n_labels)
        df_classes = pd.DataFrame(converted_trace, columns = ['class'])

    df_classified = pd.concat([df_a_subject_index, df_classes], axis=1)

    traces = extract_traces(df_classified)    
    goals = list(df_classified["Loc"].unique())
    goals.sort()
    subtraces = convert_labels_kmeans(traces, goals, subject_id)


    ########## Run the experiments with PM-based recognizers #########
    # clean the result and feedback
    os.system("rm -rf %s" % output_results)
    os.system("rm -rf Feedback")

    # number of traces for this goal (the number of traces for each goal must be the same)
    num_traces = len(subtraces[0])  # subtraces[0] = goal_0

    
    for test_id in range(num_traces): #num_traces
        reCreateDir("test")
        reCreateDir("model")
        
        goal_id = 0
        for a_goal in subtraces:
            reCreateDir("goal_%s" % str(goal_id) )
            trace_id = 0
            for a_trace in a_goal:
                write_plan(a_trace, "goal_%s/sas_plan.%s" % (str(goal_id), str(trace_id)) )
                trace_id += 1

            # test
            os.system("mv goal_%s/sas_plan.%s test/sas_plan.%s" % (str(goal_id), str(test_id), str(goal_id)) )
            # model
            os.system("java -jar sas2xes.jar goal_%s model/%s.xes" % (str(goal_id), str(goal_id)) )
            goal_id += 1
            
        # mine pnml
        os.chdir("./miningPNMLS")
        os.system("java -jar mine_all_pnmls.jar -DFM ../model/ 0.8")
        os.chdir("../")
        
        # test for all goals in this iteration
        for g in range(len(subtraces)):
            # goal_id, goal_id, percentage
            for percentage in [0.1,0.3,0.5,0.7,1.0]:
                os.system("java -jar recognizer.jar -w model/ test/sas_plan.%s %s %s %s %s %s %s %s" 
                          %(str(g), str(g), str(percentage), str(phi), str(lamb), str(delta), str(theta), str(output_results) )   )  


    result_data = pd.read_csv(output_results, usecols=[0,1,2,3,4])
    p, r, f1, a, bacc = calculate_statistics(result_data)


    return statistics.mean(p), statistics.mean(r), statistics.mean(f1), statistics.mean(a), statistics.mean(bacc)



##############################################################################
phi = 50
lamb = 1.5
delta = 1.0
theta = 0.9
subject_id = 10
n_features = 28
n_clusters = 60

run_system(phi, lamb, delta, theta, subject_id, n_features, n_clusters)

# sub_id = [1,2,3,4,5,6,7,8,9,10]
# features_num_list = [32,40,30,30,34,28,25,34,30,28]
# cluster_num_list = [60,100,180,170,100,90,100,170,140,130]

# delta_list  = [2.1870093342264347,
#      1.084800816609837,
#      0.187433151957276,
#      4.903413283459355,
#      1.8268388202335677,
#      1.0285886916143063,
#      1.2328490932305418,
#      1.849281745915097,
#      1.116650427635112,
#      0.4637984255291138]
# lamb_list  = [1.3700920003397572,
#      1.2473368231284994,
#      1.2879923507839184,
#      3.1741078302673125,
#      2.7417285189261644,
#      2.893389785116266,
#      2.1513950628675347,
#      1.0438710107770883,
#      1.1235476464151517,
#      3.0711501501373544]
# phi_list = [87, 91, 36, 16, 93, 73, 91, 46, 7, 5]
# theta_list  = [0.8220853860399499,
#      0.9165971414004777,
#      0.988840857790459,
#      0.6164212914641611,
#      0.7478471891757215,
#      0.8470576840770996,
#      0.8826274473886883,
#      0.6916758555951421,
#      0.7967492629159637,
#      0.7992781751948667]

# for i in range(10):

#     subject_id = sub_id[i]
#     n_features = features_num_list[i]
#     n_clusters = cluster_num_list[i]

#     # phi = phi_list[i]
#     # lamb = delta_list[i]
#     # delta = lamb_list[i]
#     # theta = theta_list[i]

#     phi = 50
#     lamb = 1.5
#     delta = 1.0
#     theta = 0.9

#     run_system(phi, lamb, delta, theta, subject_id, n_features, n_clusters)







