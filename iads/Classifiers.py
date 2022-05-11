# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import pandas as pd
import copy
import math
import sys
import random

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        correct_pred = 0
        for i in range(len(desc_set)):
            correct_pred+=1 if self.predict(desc_set[i])==label_set[i] else 0
        return correct_pred/len(desc_set)

# ---------------------------

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimenstion = input_dimension
        self.k = k
        
    def distance(self, x1, x2):
        return np.dot(x1-x2, x1-x2)

    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        score = 0
        # tableau des distances
        dist = np.asarray([self.distance(x,y) for y in self.desc_set])
        for i in np.argsort(dist)[:self.k]:
            score += 1 if self.label_set[i] == +1 else 0
        return 2 * (score/self.k -.5)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return -1 if self.score(x) <= 0 else 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        self.desc_set = desc_set
        self.label_set = label_set

class ClassifierKNN_MC(Classifier):
    """
    Classifieur KNN multi-classe
    """

    def __init__(self, input_dimension, k, nb_class):
        """
        :param input_dimension (int) : dimension d'entrée des exemples
        :param k (int) : nombre de voisins à considérer
        :param nb_class (int): nombre de classes
        Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
        self.nb_class = nb_class
        self.data_set = None
        self.label_set = None

    def train(self, data_set, label_set):
        self.data_set = data_set
        self.label_set = label_set

    def score(self, x):
        dist = np.linalg.norm(self.data_set-x, axis=1)
        argsort = np.argsort(dist)
        classes = self.label_set[argsort[:self.k]]
        uniques, counts = np.unique(classes, return_counts=True)
        return uniques[np.argmax(counts)]/self.nb_class

    def predict(self, x):
        return self.score(x)*self.nb_class

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        v = np.random.uniform(-1,1,self.input_dimension)
        self.w = np.random.uniform(-1,1,self.input_dimension) / np.linalg.norm(v)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """       
        #print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w,x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if (init == 0):
            self.w = np.zeros(input_dimension)
        elif (init == 1):
            self.w = 0.001 * ((2 * np.random.uniform(0, 1, input_dimension) - 1))
        else:
            return -1
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        for i in (index_list):
            Xi, Yi = desc_set[i], label_set[i]
            y_hat = np.dot(self.w, Xi)
            if (y_hat*Yi<=0):
                # self.w += self.learning_rate*np.dot(Xi,Yi)
                np.add(self.w, self.learning_rate*Yi*Xi, out=self.w, casting="unsafe")
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """ 
        self.desc_set, self.label_set = desc_set, label_set
        normes_diff_list = []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1
# ---------------------------

# CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")

class KernelBias(Kernel):
    """
    Classe pour un noyau simple 2D -> 3D
    Rajoute une colonne à 1
    """
    def transform(self, V):
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj

class KernelPoly(Kernel):
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6            
            ...
        """
        V_proj = np.hstack((np.ones((len(V),1)), V))
        t1 = V[:,0].reshape(len(V), 1)
        V_proj = np.hstack((V_proj, t1*t1))
        t2 = V[:,1].reshape(len(V), 1)
        V_proj = np.hstack((V_proj, t2*t2))
        V_proj = np.hstack((V_proj, t1*t2))
        return V_proj

class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.noyau = noyau
        if (init == 0):
            self.w = np.zeros(input_dimension)
        elif (init == 1):
            v = np.random.uniform(0, 1, input_dimension)
            v = (2*v - 1)*0.001
            self.w = v.copy()
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        # Transformation des données en dimension 6
        desc_set_transformed = self.noyau.transform(desc_set)
        for i in (index_list):
            Xi, Yi = desc_set_transformed[i,:], label_set[i]
            y_hat = np.dot(Xi, self.w)
            if (y_hat*Yi<=0):    # Il y a erreur, donc correction
                self.w += self.learning_rate*np.dot(Xi,Yi)
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        normes_diff_list, accuracy_list = [], []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            accuracy_list.append(self.accuracy(desc_set, label_set))
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list, accuracy_list
    
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        a = len(x)
        x_hat = x.reshape(1,a)
        x_hat = self.noyau.transform(x_hat)
        return np.vdot(self.w, x_hat)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        return -1 if self.score(x)<=0 else +1

# ---------------------------

class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.allw = []
        if (init == 0): self.w = np.zeros(input_dimension)
        elif (init == 1): self.w = 0.001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.allw.append(self.w.copy())
        
    def get_allw(self):
        return self.allw
    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        for i in (index_list):
            Xi, Yi = desc_set[i,:], label_set[i]
            y_hat = np.dot(self.w, Xi)
            if (y_hat*Yi<1):    # Il y a erreur, donc correction
                self.w += self.learning_rate*np.dot(Xi,Yi)
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """ 
        self.desc_set, self.label_set = desc_set, label_set
        normes_diff_list = []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

# Perceptron Multi classe
class PerceptronMultiOOA(Classifier):
    """ Perceptron multiclass OneVsAll
    """
    def __init__(self, input_dimension, learning_rate,nbC=10 , init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - nbC : nombre de classes du dataset
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        # Liste de 10 classifieurs perceptrons: 1 pour chaque classe
        self.classifieurs = [ClassifierPerceptron(input_dimension, learning_rate) for i in range(10)]

    def train(self, desc_set, label_set):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        for i in range(len(self.classifieurs)):
            # print('Iteration {}'.format(i))
            ytmp = np.where(label_set==i, 1, -1)
            self.classifieurs[i].train(desc_set,ytmp)
    
    def score(self,x):
        """ rend le score de prédiction de chaque classifieur sur x (valeur réelle)
            x: une description
        """
        a = []
        for i in range(len(self.classifieurs)):
            a.append(self.classifieurs[i].score(x))
        return a
    
    def predict(self, x):
        """ rend la prediction (classe) sur x de 0:9
            x: une description
        """
        return np.argmax(self.score(x))
    
    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


# -----------------------------------------------------------------
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.niter_max = niter_max
        self.history = history
        self.allw = []
        # Initialisation de w aléatoire
        self.w = 0.001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.allw.append(self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        seuil=0.001
        for k in range(self.niter_max):
            index_list =[i for i in range(len(desc_set))]
            np.random.shuffle(index_list)
            for i in (index_list):
                Xi, Yi = desc_set[i,:], label_set[i]
                grad = np.dot(Xi.T,np.dot(Xi,self.w)-Yi)
                w0 = self.w.copy()
                self.w -= self.learning_rate*grad
                self.allw.append(self.w.copy())
            difference = np.linalg.norm(w0-self.w)
            if (difference<seuil):
                break
    
    def get_allw(self):
        return self.allw if self.history else []
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE2
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # code de la classe ADALINE Analytique
        alpha = 1e-5
        a = desc_set.T@desc_set + alpha*np.eye(desc_set.shape[1])
        b = desc_set.T@label_set
        self.w = np.linalg.solve(a,b)
    
    def get_allw(self):
        return self.allw if self.history else []
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else 1

# Arbres de décision

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    u, c = np.unique(Y, return_counts=True)
    return u[np.argmax(c)]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    np.seterr(divide='ignore', invalid='ignore')
    return np.sum(P * -np.nan_to_num(np.log(P)))


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    u, c = np.unique(Y, return_counts=True)
    P = c / len(Y)
    return shannon(P)


class Noeud:
    """ Classe de base des noeuds des arbres de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att
        if (nom == ''): 
            nom = 'att_'+str(num_att)
        self.nom_attribut = nom 
        self.Les_fils = None
        self.classe   = None
    
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (Noeud) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = fils

    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        raise NotImplementedError

    def to_graph(self, g, prefixe='A'):
        raise NotImplementedError


class NoeudCategoriel(Noeud):
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
        
    def classifie(self, exemple):
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        for i in range(nb_col):
            gain = 0
            U, C = np.unique(X[:, i], return_counts=True)
            P = C / np.sum(C)
            for j in range(len(U)):
                uuY, cuY = np.unique(Y[np.where(X[:,i] == U[j])[0]], return_counts=True)
                puY = cuY / np.sum(cuY)
                gain += P[j] * shannon(puY)
            gain = entropie_classe - gain
            if gain > gain_max:
                gain_max = gain
                i_best = i
                Xbest_valeurs = U
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


class NoeudNumerique(Noeud):
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        Noeud.__init__(self, num_att, nom)
        self.seuil = None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
            
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        return self.Les_fils['sup'].classifie(exemple)
          
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


def partitionne(m_desc, m_class, n, s):
    w = np.where(m_desc[:,n] <= s)[0]
    left_desc, left_class = m_desc[w], m_class[w]
    w = np.where(m_desc[:,n] > s)[0]
    right_desc, right_class = m_desc[w], m_class[w]
    return ((left_desc, left_class), (right_desc, right_class))


def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        
        for i in range(nb_col):
            gain = 0
            ((seuil, entropy), (_, _)) = discretise(X, Y, i)
            partition = ((X, Y), (None, None))
            if seuil is not None:
                partition = partitionne(X, Y, i, seuil)
            gain = entropie_classe - entropy
            if gain > gain_max:
                gain_max = gain
                i_best = i
                Xbest_tuple = partition
                Xbest_seuil = seuil
        
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil,
                              construit_AD_num(left_data,left_class, epsilon, LNoms),
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

def tirage(VX, m, r = False):
    """
    Arguments: 
        VX : ensemble des indices des exemples de la BA X
        m  : taille du vecteur d'indices à rendre
        r  : booleen indiquant si il y a remise ou pas dans le tirage
    """
    return [random.choice(VX) for i in range(m)] if r else random.sample(VX,m)

def echantillonLS(LS, m, r):
    """
    Arguments: 
        LS : couple de 2 np.arrays (X,Y)
        m  : taille du vecteur d'indices à rendre
        r  : booleen indiquant si il y a remise ou pas dans le tirage
    """
    X,Y = LS[0], LS[1]
    VX = tirage([i for i in range(len(X))], m, r)
    return X[VX], Y[VX]

class ClassifierBaggingTree(Classifier):
    def __init__(self, B, perc, seuil, r):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.B = B
        self.perc = perc
        self.r = r
        self.seuil = seuil
        
    def train(self, LabeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.treeSet = set(ClassifierArbreNumerique(LabeledSet[0][1], self.seuil) for i in range(self.B))
        taille_ech = round(len(LabeledSet[0])*self.perc)
        for tree in self.treeSet:
            X, Y = echantillonLS(LabeledSet, taille_ech, self.r)
            tree.train(X,Y)
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        somme=0
        for tree in self.treeSet:
            somme += tree.predict(x)
        return +1 if somme>=0 else -1

class ClassifierRandomForest(Classifier):
    def __init__(self, B, perc, seuil, r):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.B = B
        self.perc = perc
        self.r = r
        self.seuil = seuil
        
    def train(self, LabeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.treeSet = set(ClassifierArbreNumerique(LabeledSet[0][1], self.seuil) for i in range(self.B))
        taille_ech = round(len(LabeledSet[0])*self.perc)
        for tree in self.treeSet:
            X, Y = echantillonLS(LabeledSet, taille_ech, self.r)
            tree.train(X,Y)
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        somme=0
        for tree in self.treeSet:
            somme += tree.predict(x)
        return +1 if somme>=0 else -1