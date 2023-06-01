import os
import shutil
import sys
from pm_recognizer_gene_models import run_system


selected_num_features = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
clusters_num = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]


# Define the folder name
sub = int(sys.argv[1])
folder = "sub" + str(sub)

# Check if the folder already exists
if os.path.exists(folder):
    # If it exists, delete the folder and its contents
    shutil.rmtree(folder)

# Create a new folder
os.mkdir(folder)


phi = 50
lamb = 1.5
delta = 1.0
theta = 0.9

# Kmeans:
for fn in selected_num_features:
    for cl in clusters_num:
        run_system(phi, lamb, delta, theta, sub, fn, cl)

