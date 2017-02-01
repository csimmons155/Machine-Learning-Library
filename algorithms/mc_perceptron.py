__author__ = 'christopher simmons'
import numpy as np
from classes import Predictor

class MC_Perceptron(Predictor):
    def __init__(self, instances, high_idx):
        self.instances = instances
        self.high_idx = high_idx
        self.labels = []
        self.get_labels()
        self.weights = {k: [0 for _ in range(self.high_idx)] for k in self.labels}


    def train(self, instances):
        for instance in instances:
            input_x = self.get_feat_vec(instance._feature_vector.features)
            actual_lab = instance._label.label
            arg_max, pred_lab = 0, self.labels[0]
            for c in self.labels:
                act = np.dot(input_x, self.weights[c])
                if act > arg_max:
                    arg_max, pred_lab = act, c

            #print "Best Label", pred_lab, "Arg_Max", arg_max, "Actual", actual_lab
            #quit()
            if not(actual_lab == pred_lab):
                self.weights[actual_lab] = np.add(self.weights[actual_lab], input_x)
                neg = np.dot(-1, input_x)
                self.weights[pred_lab] = np.add(self.weights[pred_lab], neg)


    def predict(self, instance):
        #print "Weights", self.weights, "updated"
        #quit()
        input_x = self.get_feat_vec(instance._feature_vector.features)
        actual_lab = instance._label.label

        arg_max, pred_lab = 0, self.labels[0]
        for c in self.labels:
            act = np.dot(input_x, self.weights[c])
            if act >= arg_max:
                arg_max, pred_lab = act, c

        #print "pred_lab", pred_lab, "actual", actual_lab
        #quit()

        return pred_lab



    def get_labels(self):
        for instance in self.instances:
            actual_lab = instance._label.label
            if actual_lab not in self.labels: self.labels.append(actual_lab)

        self.labels = sorted(self.labels)
        #print "Labels", self.labels
        #quit()


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x