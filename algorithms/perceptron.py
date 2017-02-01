__author__ = 'christopher simmons'
import numpy as np
from classes import Predictor
from scipy.sparse import csr_matrix

"""
Standard Perceptron

"""

class Perceptron(Predictor):
    def __init__(self, instances, high_idx, learn_rate):
        #instances = a list
        self.instances = instances
        self.high_idx = high_idx
        self.weights = np.array([0 for i in range(self.high_idx*5)])
        self.learn_rate = learn_rate

    def train(self, instances):
        for instance in self.instances:
            actual_lab = instance._label.label
            input_x = self.get_feat_vec(instance._feature_vector.features)

            result = np.dot(self.weights, input_x)
            if(result >= 0): est_lab = 1
            else: est_lab = 0

            if(actual_lab != est_lab):
                error = actual_lab - est_lab
                prod1 = self.learn_rate*error
                prod2 = np.dot(prod1,input_x)
                #self.weights += prod2
                self.weights = np.add(self.weights,prod2)


    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        result = np.dot(self.weights, input_x)
        if(result >= 0): est_lab = 1
        else: est_lab = 0

        return est_lab


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx*5)]
        for i in feat_vec:
            try:
                x[i-1] = feat_vec[i]
            except IndexError:
                print "Error ", i, len(x), self.high_idx*5
                quit()
        return x

"""
    Margin Perceptron
"""
class PerceptronMargin(Predictor):
    def __init__(self, instances, high_idx, learn_rate,iterate):
        #instances = a list
        self.iterate = iterate
        self.instances = instances
        self.high_idx = high_idx
        self.weights = np.array([0 for i in range(high_idx*5)])
        self.learn_rate = learn_rate


    def train(self, instances):
        for instance in instances:
            actual_lab = instance._label.label
            if actual_lab == 0: actual_lab = -1
            input_x = self.get_feat_vec(instance._feature_vector.features)
            result = np.dot(self.weights,input_x)
            prod1 = np.dot(actual_lab,result)
            if (prod1 < 1):
                increment = self.learn_rate*actual_lab
                increment = np.dot(increment,input_x)
                self.weights = np.add(self.weights,increment)

    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        result = np.dot(self.weights,input_x)

        if(result >= 0): est_lab = 1
        else: est_lab = 0

        return est_lab


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx*5)]
        #on train file the starting idx = 1, not 0
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x



"""
Average Perceptron 

"""
class AveragePerceptron(Predictor):
    def __init__(self, instances, high_idx, learn_rate):
        #instances = a list
        self.instances = instances
        self.high_idx = high_idx
        self.weights = np.array([0 for i in range(high_idx*5)])
        self.sum_weights = np.array([0 for i in range(high_idx*5)])
        self.count = 0
        self.learn_rate = learn_rate

    def train(self, instances):
        for instance in self.instances:
            actual_lab = instance._label.label
            input_x = self.get_feat_vec(instance._feature_vector.features)
            result = np.dot(self.weights, input_x)
            if(result >= 0): est_lab = 1
            else: est_lab = 0

            if(actual_lab != est_lab):
                error = actual_lab - est_lab
                prod1 = self.learn_rate*error
                prod2 = np.dot(prod1,input_x)
                #self.weights += prod2
                self.weights = np.add(self.weights,prod2)
                #self.sum_weights += self.weights
                self.sum_weights = np.add(self.sum_weights,self.weights)
                self.count += 1
            else:
                #self.sum_weights += self.weights
                self.sum_weights = np.add(self.sum_weights,self.weights)
                self.count += 1

    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        weight_final = self.sum_weights / self.count

        result = np.dot(weight_final, input_x)
        if(result >= 0): est_lab = 1
        else: est_lab = 0

        return est_lab

    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx*5)]
        for i in feat_vec: x[i-1] = feat_vec[i]

        return x
