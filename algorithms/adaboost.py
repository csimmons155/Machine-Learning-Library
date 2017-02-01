__author__ = 'christopher simmons'
import numpy as np
from classes import Predictor
from scipy.sparse import csr_matrix
import math
import operator

class Adaboost(Predictor):
    """
       Adaboost Classifier
        
    """
    def __init__(self,instances,T,high_idx):
        self.instances = instances
        self.high_idx = high_idx
        # list of xi's, number of features, number of xi's, list of labels
        self.collection_x, self.j_size, self.total_samples, self.collection_y = self.get_coll(self.instances)

        self.c_j_tuples = self.get_tuples(self.collection_x,self.j_size)

        #number of iterations/rounds
        self.T = T

        #list of the best predictors [the (j,c) tuples] for every t from 1 to T iterations
        self.predict_list = []


    def get_coll(self,instances):
        """
        Get a collection of xi,yi input values and store them in seperate lists
        :param instances:
        :return:
        """
        ret_list = []
        ret_lab = []
        count = 0
        length = 0
        #print "Getting xi's"
        for instance in instances:
            input_x = self.get_feat_vec(instance._feature_vector.features)
            if len(input_x) > length: length = len(input_x)
            actual_lab = instance._label.label
            count += 1
            #same order as in the instances list of xi,yi
            ret_list.append(input_x)
            ret_lab.append(actual_lab)

        #print "Gottem'"
        return ret_list, length, count, ret_lab

    def get_tuples(self, coll, j_size):
        """
        Store tuples of jth index of xi and the corresponding c value.
        Both make up our weak predictor
        :param coll: list of xi values
        :param j_size: the length of all the xi vectors
        :return: list of (j,c) tuples
        """

        tuple_list = []
        #for every feature: j
        #print "Getting (j,c) tuples"
        for j in range(j_size):
            list_of_js = []
            for i in range(len(coll) - 1):
                if coll[i][j] == 0:
                    continue
                list_of_js.append(coll[i][j])

                #print "Should be Non_Zero", coll[i][j]
            list_of_js = sorted(list_of_js)
            #t = 0
            for k in range(len(list_of_js) - 1):
                #c = (coll[i+1][j] + coll[i][j])/2.0
                c = (list_of_js[k+1] + list_of_js[k])/2.0
                #t += 1
                #print "Number of distinct c's for j = ", j," ",t,"/", self.total_samples
                if (j,c) in tuple_list:
                    continue
                else:
                    tuple_list.append((j,c))

        #print "Got those"
        #print tuple_list
        return tuple_list

    def weakLearner2(self, tuple, xi):
        j,c = tuple
        if xi[j] > c:
            est_lab = 1
        else:
            est_lab = -1

        return est_lab



    def weakLearner(self, tuple, xi):
        """
        The weak learner
        :param tuple: the (j,c) value
        :param xi: the input vector
        :return: the estimated label for the xi input
        """

        j,c = tuple
        scores = {1:0, -1:0}
        xi_list = self.collection_x
        #print "Tuple (j,c) ", j, c, xi[j]
        if xi[j] > c:
            for k in range(len(xi_list)):
                if xi_list[k] > c:
                    if self.collection_y[k] == 1:
                        scores[1] += 1
                    else:
                        scores[-1] += 1

            order = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
            return order[0][0]

        else:
            for k in range(len(xi_list)):
                if xi_list[k] <= c:
                    if self.collection_y[k] == 1:
                        scores[1] += 1
                    else:
                        scores[-1] += 1

            order = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
            #print "Done <= c"
            return order[0][0]


    def get_alpha(self,error):
        """
        Get the alpha value
        :param error:
        :return: alpha value
        """
        num = 1 - error
        return 0.5*math.log((num/error))



    def train(self, instances):
        #initialize distribution
        D = [1.0/self.total_samples] * self.total_samples
        mid_D = [1.0/self.total_samples] * self.total_samples

        error = 0
        min_error = 100000
        for t in range(self.T):
            #print "Train", t
            min_error = 100000
            error = 0
            for tuple in self.c_j_tuples:
                #the number of xi's
                #print "Our tuple: ", tuple
                for i in range(len(self.collection_x)):
                    act_lab = self.collection_y[i]
                    if act_lab == 0: act_lab = -1
                    est_lab = self.weakLearner2(tuple,self.collection_x[i])
                    #print "Estimated:", est_lab, "Actual", act_lab
                    if act_lab != est_lab:
                        error += D[i]

                #print "Our error: ", error
                if error < 0.000001:
                    return
                if error < min_error:
                    min_error = error
                    min_tuple = tuple
                    #print "The new min error =", min_error, "for this tuple: ", min_tuple


            #print "Getting alpha"
            #found our optimal hypothesis by now
            alpha = self.get_alpha(min_error)
            #print "Found alpha = ", alpha
            Z_sum = 0

            for k in range(len(self.collection_x)):
                h_t = self.weakLearner2(min_tuple,self.collection_x[k])
                y_i = self.collection_y[k]
                if y_i == 0: y_i = -1
                exp = -(alpha*y_i*h_t)
                Z_sum += (D[k]*math.exp(exp))
                mid_D[k] = D[k]*math.exp(exp)
                #print "Traverse to get alpha stuff", k

            #print "Appending to predict_list"
            self.predict_list.append((alpha,min_tuple))

            #get our new distribution
            for z in range(len(mid_D)):
                D[z] = (D[z]*mid_D[z])/Z_sum
                #D[z] = (mid_D[z])/Z_sum


    def predict(self, instance):
        #print "Predicting..."
        sum = 0
        actual_lab = instance._label.label
        if actual_lab == 0: actual_lab = -1
        input_x = self.get_feat_vec(instance._feature_vector.features)

        for i in range(self.T):
            alpha,tuple = self.predict_list[i]
            est_lab = self.weakLearner(tuple,input_x)
            if est_lab == actual_lab: multi = 1
            else: multi = 0
            sum += alpha*multi

        #print "Sum :", sum, "actual_lab", actual_lab, "(if sum > 0, then label is 1)"

        if sum > 0:
            return 1
        else:
            return 0


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x