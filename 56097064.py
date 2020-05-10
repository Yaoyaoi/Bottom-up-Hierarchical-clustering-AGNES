import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matrix = []
nodes = []
k = 0

# loadData
def loadData(matrixFileName):
    global matrix, nodes, k
    fid = open(matrixFileName)
    lines = fid.readlines()
    nodes = lines.pop(0).split()
    k = len(nodes)
    for i in range(k):
        lineO = lines[i].split()
        line = [0 for j in range(i+1)]
        line.extend([lineO[j+1] for j in range(i+1,k)])
        line = list(map(float,line))
        matrix.append(line)
    matrix = np.array(matrix)
    matrixT = matrix.T
    matrix = matrix + matrixT

#-----------START------------
# cluster
class Cluster():
    def __init__(self,left=None,right=None,simi=1,sList=None):
        """
        :param left: left cluster
        :param right:  right cluster
        :param sList: all the nodes in this cluster
        :param simi: similarity between left cluster and right cluster
        :param count: number of nodes in this cluster
        """
        self.left = left
        self.right = right
        self.simi = simi
        self.count = 1
        self.sList = sList
        if left:
            self.sList = left.sList[:]
            self.sList.extend(right.sList)
            self.count = left.count + right.count

# Hierachical bottom to up clustering
class HierarchicalBUClustering():
    def __init__(self,matrix,k):
        """
        :param matrix: similarity matrix
        :param k: number of nodes
        """
        self.omatrix = matrix
        self.k = k
        self.clusters = [Cluster(sList=[i]) for i in range(k)]    # make each sample a cluster
        self.matrix = matrix
        self.tree = None
    
    #calculate the average similarity of two clusters
    def calculateAVGSimilarity(self, c1, c2):
        """
        :param c1,c2: cluster1 and cluster2 to be calculated
        """
        sumOfSimi = 0
        for node1 in c1.sList:
            for node2 in c2.sList:
                sumOfSimi += self.omatrix[node1,node2]
        avgS = sumOfSimi / (c1.count * c2.count)
        return avgS

    # iteration
    def fit(self, kList=None):
        """
        :param kList: the list of k values that you what to save.
        :return self.tree: the Final tree
        :return kClustersList: the list of (k,clusterList), where k is the values in the kList you give
        """
        kClustersList = []
        while self.k > 1:
            if self.k in kList:
                kClustersList.append((self.k,self.clusters[:]))
            # find the max similarity in current matrix
            c1Index = np.where(self.matrix == np.max(self.matrix))[0][0]
            c2Index = np.where(self.matrix == np.max(self.matrix))[1][0]
            c1 = self.clusters[c1Index]
            c2 = self.clusters[c2Index]
            # merge cluster1 and cluster2
            c3 = Cluster(c1,c2,np.max(self.matrix))
            # delete
            # make sure the bigger one first delete
            if c1Index < c2Index:
                temp = c1Index
                c1Index = c2Index
                c2Index = temp
            self.clusters.pop(c1Index)
            self.clusters.pop(c2Index)
            self.matrix = np.delete(self.matrix,(c1Index,c2Index),0)
            self.matrix = np.delete(self.matrix,(c1Index,c2Index),1)
            # add
            if self.clusters:
                c3Simi = [self.calculateAVGSimilarity(c,c3) for c in self.clusters]
                self.matrix = np.vstack((self.matrix,c3Simi))
                c3Simi.append(0)
                self.matrix = np.hstack((self.matrix,np.array([c3Simi]).T))
            self.clusters.append(c3)
            self.k = len(self.clusters)
        self.tree = self.clusters[0]
        if self.k in kList:
            kClustersList.append((self.k,self.clusters[:]))
        return self.tree,kClustersList
#------------END-------------

# print Senquence IDs for k = 2 to 5
def printSenquenceIDs(kClustersList,nodes):
    for kClusters in kClustersList:
        print(kClusters[0])
        for cluster in kClusters[1]:
            for sID in cluster.sList:
                print(nodes[sID],end=' ')
            print()

# plot hotmap for k = 2 to 5
def plotHotmap(kClustersList,matrix):
    for i in range(len(kClustersList)):
        cMatrix = []
        kClusters =  kClustersList[i]
        plt.figure(i)
        plt.title(kClusters[0])
        nodeList = []
        for c in kClusters[1]:
            nodeList.extend(c.sList)
        for node1 in nodeList:
            simiList = []
            for node2 in nodeList:
                simiList.append(matrix[node1][node2])
            cMatrix.append(simiList)
        sns.heatmap(cMatrix, cmap = 'seismic')
    plt.show()


# main part
if __name__ == "__main__":
    fileName = input("input the file'path:")
    loadData(fileName)
    clustering = HierarchicalBUClustering(matrix,k)
    kList = [2,3,4,5]
    finalTree,kClustersList = clustering.fit(kList)
    printSenquenceIDs(kClustersList,nodes)
    plotHotmap(kClustersList,matrix+np.identity(k))