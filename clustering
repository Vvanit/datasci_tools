import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#===== main function
def fit_kmeans(X_scale,model=KMeans,min_cluster = 3,max_cluster = 7):
    X_cut = X_scale
    step = []
    while True:
        X_cut=X_cut
        plot_elbow(X_cut,model=model,min_cluster = min_cluster,max_cluster = max_cluster+1)
        while True:
            try:
                n_clusters = int(input("Please input number of cluster from {0} to {1}: \n".format(min_cluster,max_cluster)))
                if n_clusters not in range(min_cluster,max_cluster+1):
                    print('Value error')
                    continue
                else:
                    step.append(n_clusters)
                    kmeans = model(n_clusters=n_clusters,n_init=100)
                    kmeans.fit(X_cut)
                    cluster = kmeans.predict(X_cut)
                    cut_list = find_cut_list(cluster)
                    print('Number of member of {} clusters is'.format(n_clusters))
                    print(list(zip(np.unique(cluster),np.bincount(cluster))))
                    break
            except ValueError:
                print('Value error')
                continue
        if cut_list!=[]:
            print('purpose cut cluster in list',cut_list)
            res = input("Input 'Y' to cut cluster 'N' to get cluster result:\n")
            if res.lower()=='n':
                cluster = kmeans.predict(X_scale)
                print('Clustering complete!!!')
                break
            elif res.lower=='y':
                cluster_filter = []
                for c in cluster:
                    if c in cut_list: #build filter array
                        cluster_filter.append(False)
                    else:
                        cluster_filter.append(True)
                X_cut = X_cut[np.array(cluster_filter)]
                continue
            else:
                continue
                
        else:
            cluster = kmeans.predict(X_scale)
            print('Clustering complete!!!')
            break
    return cluster,kmeans,step #return cluster result and kmeans model

#===== sub finction =====
def plot_elbow(X_scaled,model = KMeans,min_cluster = 3,max_cluster = 8):
    wcss=[]
#     fig = plt.figure()
    for i in range(min_cluster,max_cluster):
        kmeans = model(i)
        kmeans.fit(X_scaled)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    number_clusters = range(min_cluster,max_cluster)
    plt.plot(number_clusters,wcss)
    # plt.title('The Elbow title')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    plt.show()
    return

def find_cut_list(cluster,thersold = 0.05):
    cut_list = []
#     thersold = 0.05
    for c in list(zip(np.unique(cluster),np.bincount(cluster))):
        if c[1]==1 or c[1]/len(cluster)<thersold:
            cut_list.append(c[0])
    return cut_list

