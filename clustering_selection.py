
import numpy as np
import pandas as pd
import pydpc
from pydpc import Cluster


def clustering_dpc(X_train_majority,X_train_minority,y_train_majority,y_train_minority,num1,num2):
   
    dpc = Cluster(X_train_majority,fraction=0.001,autoplot=False)
    dpc1 = Cluster(X_train_majority,fraction=0.001,autoplot=False)
    delt=dpc1.delta
    dens=dpc1.density
    
    delt.sort()
    dens.sort()
    n=len(delt)
    d=0
    
    while d<n-1:
        distance1= delt[d+1]-delt[d]
        if distance1 < 0.1:
           d=d+1
        else:
           distance1=delt[d]+0.01
           d=n+10

    d=0     
    while d<n-1:
        distance2= dens[d+1]-dens[d]
        if distance2 < 0.01:
           d=d+1
        else:
           distance2=dens[d]+0.01
           d=n+10

    dpc.assign(distance2,distance1)

    num_clusters = dpc.clusters.shape[0]
    
    def compute_cluster_instances_index(number_of_clusters): #index of instances of each cluster
        cluster_index = []
        for num in range(num_clusters):
            cluster_index.append(np.where(dpc.membership==num)[0])
        return cluster_index
    
    def compute_cluster_instances_density(cluster_index,number_of_clusters): # list of densities of each instance 
        cluster_instances_density = []
        for num in range(num_clusters):
            cluster_instances_density.append(dpc.density[cluster_index[num]])
        return cluster_instances_density

    def whole_density_of_cluster(cluster_instances_density):
        cluster_density = []
        for i in cluster_instances_density:
            cluster_density.append(sum(x for x in i))
        return cluster_density
   
    def centroid_distance_from_minority(centroids): #distance of each centroid from whole minority instances
        cluster_dist = []
        for c in centroids:
            dist = 0.0
            for i in range(X_train_minority.shape[0]):
                dist += np.linalg.norm(X_train_majority[c]-X_train_minority[i])
            cluster_dist.append(dist)
        return cluster_dist
    cluster_index = compute_cluster_instances_index(num_clusters)
    cluster_ins_den = compute_cluster_instances_density(cluster_index,num_clusters)
    clusters_density = whole_density_of_cluster(cluster_ins_den)
    cluster_dis = centroid_distance_from_minority(dpc.clusters)
    cluster_dis.sort(reverse = True)

    clusters_density_scl = clusters_density/(sum(clusters_density))
    cluster_dis_scl = cluster_dis/(sum(cluster_dis))
    
    return cluster_index,clusters_density_scl,cluster_dis_scl,cluster_ins_den

def selection(X_train_majority,X_train_minority,y_train_majority,y_train_minority,cluster_index,
              clusters_density,cluster_distance,alpha,beta,cluster_ins_den):
    num_clusters = len(clusters_density)

    z = alpha * clusters_density + beta * cluster_distance
    
    number_of_selected = np.zeros(z.shape)
    list1 = []
    for i in range(num_clusters):
        list1.append(round((z[i] * (X_train_minority.shape[0]*6)) / sum(z)))
        
        if list1[i] > cluster_index[i].shape[0]:
            number_of_selected[i] = cluster_index[i].shape[0]
        else:
            number_of_selected[i] = round(list1[i])

    number_of_selected_final = number_of_selected.astype(int)
    #print("number of selected instances from all clusters:", sum(number_of_selected_final))
    final_list = []
    for i, j in enumerate(number_of_selected_final): 
            final_list.append(np.argsort(cluster_ins_den[i])[::-1][:j]) #select cluster i and sort its index
    #return final_list #list index of selected data points    
    
    flat_list = [item for sublist in final_list for item in sublist] # flat list of final list

    X_train_select = []
    y_train_select = []
    indexs = []
    for i in flat_list:
        X_train_select.append(X_train_majority[i])
        indexs.append(i)
        y_train_select.append(y_train_majority[i])
    X_train_select = np.array(X_train_select)
    y_train_select = np.array(y_train_select)
    indexs = np.array(indexs)

    y_train_minority = np.array(y_train_minority)
    X_train_balanced = np.concatenate((X_train_select,X_train_minority))
    y_train_balanced = np.concatenate((y_train_select,y_train_minority))
    return X_train_balanced, y_train_balanced, indexs
