__author__ = 'christopher simmons'

import numpy as np
from classes import Predictor
import math
import operator

class KNN(Predictor):
    """
     K Nearest Neighbors Classifier
     
    """
    def __init__(self,instances,k,high_idx):
        self.k = k
        self.instances = instances
        self.t_set = {}
        self.high_idx = high_idx

    def train(self, instances):
        """
        Create training set
        :param instances: the (features,label) pairs for training
        """
        for instance in self.instances:
            act_lab = instance._label.label
            x_input = self.get_feat_vec(instance._feature_vector.features)
            if act_lab in self.t_set:
                self.t_set[act_lab].append(x_input)
            else:
                self.t_set[act_lab] = [x_input]


    def euclideanDistance(self,inst1, inst2, length):
        dist = 0
        for i in range(length):
            dist += pow((inst2[i] - inst1[i]),2)
        return math.sqrt(dist)


    def getNeighbors(self,test_sample):
        distances = []
        length = len(test_sample)
        #for every label yi
        for i in self.t_set:
            #for every xi
            for j in self.t_set[i]:
                dist = self.euclideanDistance(test_sample, j, length)
                distances.append((dist, i))

        #sort distance from min to max
        distances = sorted(distances, key= lambda x:x[0])

        #array of labels
        neighbors = []
        for j in range(self.k):
            neighbors.append(distances[j][1])

        return neighbors


    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        #input_x goes to getNeighbors function
        near_neigh = self.getNeighbors(input_x)
        scores = {}

        for i in near_neigh:
            if i not in scores:
                scores[i] = 0
            if i == actual_lab: scores[i] += 1

        #order = sorted(scores,key=scores.get,reverse=True)
        #sort the dictionary by value
        order = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        if len(order) > 1 and order[0][1] == order[1][1]:
            return min(order[0][0],order[1][0])

        return order[0][0]


    def get_feat_vec(self,feat_vec):
        """
        :param feat_vec: dictionary of features in order
        :return: list of the features
        """
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x


class Distance_KNN(Predictor):
    def __init__(self,instances,k,high_idx):
        self.k = k
        self.instances = instances
        self.t_set = {}
        self.high_idx = high_idx

    def train(self, instances):
        """
        Create training set
        :param instances: the (features,label) pairs for training
        """
        for instance in self.instances:
            act_lab = instance._label.label
            x_input = self.get_feat_vec(instance._feature_vector.features)
            if act_lab in self.t_set:
                self.t_set[act_lab].append(x_input)
            else:
                self.t_set[act_lab] = [x_input]


    def euclideanDistance(self,inst1, inst2, length):
        dist = 0
        for i in range(length):
            dist += pow((inst1[i] - inst2[i]),2)
        return math.sqrt(dist)


    def getNeighbors(self,test_sample):
        distances = []
        length = len(test_sample) - 1
        for i in self.t_set:
            for j in self.t_set[i]:
                dist = self.euclideanDistance(test_sample,j,length)
                distances.append((dist,i,j))

        distances = sorted(distances, key= lambda x:x[0])

        #array of (label,feature_vector) tuples
        neighbors = []
        for j in range(self.k):
            neighbors.append((distances[j][1],distances[j][2]))

        return neighbors


    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        #input_x goes to getNeighbors function
        near_neigh = self.getNeighbors(input_x)
        scores = {}

        for i,j in near_neigh:
            if i not in scores:
                scores[i] = 0
            if i == actual_lab:
                dist = (self.euclideanDistance(input_x,j,len(input_x)))**2
                weight = 1/(1+dist)
                scores[i] += weight

        #order = sorted(scores,key=scores.get,reverse=True)
        #return order[0]
        order = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        if len(order) > 1 and order[0][1] == order[1][1]:
            return min(order[0][0],order[1][0])

        return order[0][0]


    def get_feat_vec(self,feat_vec):
        """
        :param feat_vec: dictionary of features in order
        :return: list of the features
        """
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x


