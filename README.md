# Bottom-up-Hierarchical-clustering-AGNES
An implement of  AGNES——bottom up hierarchical clustering
### instruction of my program
#### how to run
1. run my program by: python3 56097064.py 
2. then type the full path + name of the input sequence file, like:/documents/hw5/SCOV2_96_matrix.txt
#### environment
* python 3.6.8
* matplotlib 3.1.2
* seaborn 0.10.1

### How generate different number of clusters from the final tree
The fianl tree return by the function`fit(kList=None)`
I save the clusters in the iteration.
when you use function `fit(kList=None)` ，you can give the function an list of k that you what to get.
Then during the training process, the conditions that match the k will be saved as a list `kClustersList` and return it.