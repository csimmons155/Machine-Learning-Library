__author__ = 'christopher simmons'

import numpy as np
from classes import Predictor
import math
import sys

class Lambda_Means2(Predictor):
   
    #Lambda Means Classifier
    
    def __init__(self, instances, high_idx, lambda_clust, clust_iter):
        self.instances = instances
        self.high_idx = high_idx
        #self.lambda_clust = lambda_clust
        if lambda_clust == 0.0:
            self.lambda_clust = self.get_lambda(self.instances)
        else:
            self.lambda_clust = lambda_clust

        #k = 0
        self.mu_list = []
        first_mu = self.initial_mu(self.instances)
        self.mu_list.append(first_mu)

        #will be a list of lists:
        self.r_list = []

        #list of xi's
        self.feature_array = []
        self.k_count = 1

        self.clust_iter = clust_iter


    def get_lambda(self,instances):
        count = 0
        sum = 0
        vec = []
        for instance in instances:
            input_x = self.get_feat_vec(instance._feature_vector.features)
            vec.append(input_x)
            sum = np.add(sum, input_x)
            count += 1

        #mean vector

        avg = sum / float(count)
        sum3 = 0
        sum2 = 0
        for i in vec:
            for j in range(len(avg)):
                diff = (i[j] - avg[j])**2

                sum3 = np.add(sum3, diff)

            sum3 = math.sqrt(sum3)
            sum2 = np.add(sum2, sum3)

        lamb = sum2 / float(count)

        return lamb




    def train(self, instances):

        for i in range(self.clust_iter):
            self.r_list = []
            self.feature_array = []
            count = 0
            for instance in instances:
                x_input = self.get_feat_vec(instance._feature_vector.features)
                self.r_list.append([0]*self.k_count)
                self.feature_array.append(x_input)

                dist_array = []
                corres_clust = []

                clust_num = 0
                min_dist = sys.maxint
                corr_clust = 0

                for mu in self.mu_list:
                    total_dist = 0
                    for i in range(len(mu)):
                        dist = (x_input[i] - mu[i])**2
                        total_dist += dist
                    total_dist = math.sqrt(total_dist)
                    if (total_dist < min_dist):
                        min_dist = total_dist
                        corr_clust = clust_num
                    #dist_array.append(total_dist)
                    #corres_clust.append(clust_num)
                    clust_num += 1

                copy_dists = sorted(dist_array)

                #min_dist = 0
                """
                if len(copy_dists) > 1 and copy_dists[0] == copy_dists[1]:
                    idx1 = dist_array.index(copy_dists[0])
                    idx2 = dist_array.index(copy_dists[1])
                    if idx1 < idx2: min_dist = dist_array[idx1]
                    else: min_dist = dist_array[idx2]
                else:
                    min_dist = copy_dists[0]
                """
                #print "Min_dist", min_dist, "Lambda", self.lambda_clust
                if min_dist > self.lambda_clust:
                    #create a new cluster
                    self.mu_list.append(x_input)
                    self.r_list[count].append(1)
                    #append to all the r's
                    for i in range(len(self.r_list)):
                        if i != count: self.r_list[i].append(0)

                    count += 1
                    self.k_count += 1
                else:
                    #min_dist_idx = dist_array.index(min_dist)
                    #our_cluster = corres_clust[min_dist_idx]

                    our_cluster = corr_clust
                    self.r_list[count][our_cluster] = 1
                    sum1 = 0
                    #count_cluster = the number of xi's in our cluster k
                    count_cluster = 0
                    for i in range(len(self.r_list)):
                        if self.r_list[i][our_cluster] == 1:
                            count_cluster += 1
                            sum1 = np.add(sum1, self.feature_array[i])
                    new_mu = sum1 / float(count_cluster)
                    self.mu_list[our_cluster] = new_mu
                    count += 1

                #print "Clusters", self.k_count
                #if self.k_count > 839: break

            #print "Lambda", self.lambda_clust
        #quit()



    def predict(self, instance):

        input_x = self.get_feat_vec(instance._feature_vector.features)
        clust_num = 0
        corr_clust = 0
        #dist_array = []
        #clust_array = []
        min_dist = sys.maxint
        for mu in self.mu_list:
            sum1 = 0
            for i in range(len(mu)):
                diff = (input_x[i] - mu[i])**2
                sum1 += diff
            sum1 = math.sqrt(sum1)
            if sum1 < min_dist:
                min_dist = sum1
                corr_clust = clust_num
            clust_num += 1
            #dist_array.append(sum1)
            #clust_array.append(clust_num)
        our_cluster = corr_clust

        """
        copy_dist_array = sorted(dist_array)
        if len(dist_array) > 1 and copy_dist_array[0] == copy_dist_array[1]:
            idx1 = dist_array.index(copy_dist_array[0])
            idx2 = dist_array.index(copy_dist_array[1])
            if idx1 < idx2: min_dist = dist_array[idx1]
            else: min_dist = dist_array[idx2]
        else:
            min_dist = copy_dist_array[0]

        our_index = dist_array.index(min_dist)
        our_cluster = clust_array[our_index]
        """
        return our_cluster




    def initial_mu(self, instances):
        sum = 0
        count = 0
        for instance in instances:
            input_x = self.get_feat_vec(instance._feature_vector.features)
            sum = np.add(sum, input_x)
            count += 1
        avg = sum / float(count)
        return avg


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x

