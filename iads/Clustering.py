# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import sys
import random
from itertools import combinations
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist

def normalisation(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def dist_euclidienne(x,y):
    return np.linalg.norm(x-y)

def dist_manhattan(x,y):
    return sum(abs(val1-val2) for val1, val2 in zip(x,y))

def dist_vect(metric,x,y):
    return dist_euclidienne(x,y) if metric=='euclidienne' else dist_manhattan(x,y)

def centroide(X):
    return np.mean(X,axis=0)

def dist_centroides(X1,X2):
    return dist_vect('euclidienne',centroide(X1), centroide(X2))

def initialise(df):
    return {i:[i] for i in range(len(df))}

def fusionne(df, partition, verbose=False):
    dist_min = +np.inf
    k1_min, k2_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist= dist_centroides(df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_min, k2_min = k1, k2
    if k1_min != -1:
        del p_new[k1_min]
        del p_new[k2_min]
        p_new[max(partition)+1] = [*partition[k1_min], *partition[k2_min]]
        if verbose:
            print(f'Distance mininimale trouvée entre  [{k1_min}, {k2_min}]  =  {dist_min}')
    return p_new, k1_min, k2_min, dist_min

def clustering_hierarchique(df):
    result = []
    partition = initialise(df)
    for o in range(len(df)):
        partition,k1, k2, distance = fusionne(df, partition)
        result.append([k1, k2, distance, len(partition[max(partition.keys())])])
    return result[:-1]

def clustering_hierarchique(df,verbose=False, dendrogramme=False):
    partition = initialise(df)    
    results = []
    for o in range(len(df)):
        partition, k1, k2, dist = fusionne(df, partition, verbose=verbose)
        results.append([k1, k2, dist, len(partition[max(partition.keys())])])
    results = results[:-1]
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(
            clustering_hierarchique(df), 
            leaf_font_size=24.,
        )
        plt.show()
    return results

def dist_linkage_clusters(linkage, dist_func, arr1, arr2):
    r = cdist(arr1, arr2, dist_func)
    if linkage == 'complete':
        return np.max(r)
    if linkage == 'simple':
        return np.min(r)
    if linkage == 'average':
        return np.mean(r)
    
def fusionne_linkage(linkage, df, partition, dist_func='euclidean', verbose=False):
    dist_min = +np.inf
    k1_min, k2_min = -1, -1
    out_partition = dict(partition)
    for k1, v1 in partition.items():
        for k2, v2 in partition.items():
            if k1 == k2:
                continue
            dist = dist_linkage_clusters(linkage, dist_func, df.iloc[v1], df.iloc[v2])
            if dist < dist_min:
                dist_min = dist
                k1_min, k2_min = k1, k2
    out_partition = dict(partition)
    if k1_min != -1:
        del out_partition[k1_min]
        del out_partition[k2_min]
        out_partition[max(partition)+1] = [*partition[k1_min], *partition[k2_min]]
    if verbose:
        print(f'Distance mininimale trouvée entre  [{k1_min}, {k2_min}]  =  {dist_min}')
    return out_partition, k1_min, k2_min, dist_min

def clustering_hierarchique_linkage(linkage, df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    partition = initialise(df)
    results = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne_linkage(linkage, df, partition, 
                                                   dist_func, verbose)
        results.append([k1, k2, dist, len(partition[max(partition.keys())])])
    results = results[:-1]
    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(results, leaf_font_size=24.)
        plt.show()
    return results

def clustering_hierarchique_complete(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('complete', df, dist_func,
                                          verbose, dendrogramme)
def clustering_hierarchique_simplee(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('simple', df, dist_func,
                                          verbose, dendrogramme)
def clustering_hierarchique_average(df, dist_func='euclidean', 
                                    verbose=False, dendrogramme=False):
    return clustering_hierarchique_linkage('average', df, dist_func,
                                          verbose, dendrogramme)
'''
def dist_vect(v1, v2):
    return dist_euclidienne(v1,v2)
'''

def inertie_cluster(Ens):
    c = centroide(Ens)
    return np.sum([dist_vect(v,c)**2 for v in np.array(Ens)])

def init_kmeans(K,Ens):
    return np.array(pd.DataFrame(Ens).sample(n=K))

def plus_proche(Exe,Centres):
    return np.argmin([dist_vect(Exe,c) for c in Centres])

def affecte_cluster(Base,Centres):
    U = {i : [] for i in range(len(Centres))}
    for i in range(len(Base)):
        U[plus_proche(np.array(Base)[i],Centres)].append(i)
    return U

def nouveaux_centroides(Base,U):
    X = np.array(Base)
    result = []
    for k,desc in U.items():
        result.append(np.mean([X[i] for i in desc], axis=0))
    return np.array(result)

def inertie_globale(Base, U):
    X = np.array(Base)
    return sum([inertie_cluster([X[i] for i in desc]) for desc in U.values()])

def kmoyennes(K, Base, epsilon, iter_max):
    Centres = init_kmeans(K,Base)
    U = affecte_cluster(Base,Centres)
    for i in range(iter_max):
        inertie1 = inertie_globale(Base,U)
        Centres = nouveaux_centroides(Base,U)
        U = affecte_cluster(Base,Centres)
        diff = round(inertie1-inertie_globale(Base,U),4)
        print('Iteration {} Inertie : {} Difference : {}'.format(i+1,round(inertie_globale(Base,U),4),diff))
        if diff<epsilon: break
    return Centres, U

def affiche_resultat(Base,Centres,Affect):
    colors = ['g', 'b', 'y','c', 'm']
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
    for v in Affect.values():
        c = random.choice(colors)
        for i in v:
            plt.scatter(Base.iloc[i,0],Base.iloc[i,1],color=c)
    
def dunn_index(Centres):
    L = combinations([i for i in range(len(Centres))],2)
    dist = [dist_centroides(Centres[i],Centres[j]) for (i,j) in L]
    return round(min(dist)/max(dist),2)