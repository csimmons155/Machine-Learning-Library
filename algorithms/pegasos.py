__author__ = 'christopher simmons'
import numpy as np
from classes import Predictor
from scipy.sparse import csr_matrix


class Pegasos(Predictor):
    def __init__(self, instances, high_idx, my_lambda):
        #instances = a list
        self.instances = instances
        self.high_idx = high_idx
        self.weights = np.array([0 for i in range(high_idx*5)])
        self.my_lambda = my_lambda
        self.t = 0

    def train(self, instances):
        #t = 0
        for instance in self.instances:
            actual_lab = instance._label.label
            #change to -1
            if actual_lab == 0: actual_lab = -1
            input_x = self.get_feat_vec(instance._feature_vector.features)
            #time step
            self.t += 1
            first_coeff = 1 - 1.0/self.t
            result = np.dot(self.weights, input_x)
            prod1 = actual_lab*result
            if prod1 < 1:
                sec_coeff = 1.0/(self.my_lambda*self.t)
                prod2 = np.dot(actual_lab,input_x)
                add_term = np.dot(sec_coeff,prod2)
                f_add_term = np.dot(first_coeff,self.weights)
                self.weights = np.add(f_add_term,add_term)
            else:
                self.weights = np.dot(first_coeff,self.weights)


            """
            g_left = np.array([0 for z in range(len(self.weights))])
            for k in range(len(self.weights)): g_left[k] = self.my_lambda*self.weights[k]

            if prod1 < 1:
                g_right = np.array([0 for z in range(len(input_x))])
                for k in range(len(input_x)): g_right[k] = actual_lab*input_x[k]
                #our gradient
                gradient = g_left - g_right

            else:
                gradient = g_left

            step_size = 1 / (self.my_lambda*t)

            adj = step_size*gradient
            self.weights = self.weights - adj"""

    def predict(self, instance):
        actual_lab = instance._label.label
        input_x = self.get_feat_vec(instance._feature_vector.features)
        result = np.dot(self.weights, input_x)
        #weights = csr_matrix(self.weights)
        #x = np.array(input_x)
        #result = weights.dot(x)[0]

        if(result >= 0): est_lab = 1
        else: est_lab = 0

        return est_lab

    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx*5)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x

