__author__ = 'christopher simmons'

import numpy as np
from classes import Predictor
import math

class Naive_Bayes(Predictor):
    """
    Naive Bayes Classifier     
    """
    def __init__(self, instances, high_idx, K):
        self.instances = instances
        self.K = K
        self.high_idx = high_idx
        # dictionary: {([xi],k)}
        self.cluster_matches = []

        self.cluster_info = {}
        self.num_of_examples = 0
        for i in range(self.K):
            #each kth element is (mu_k, sigma_k, phi_k)
            self.cluster_info[i] = ([],[],0)


    def assign_clusters(self):
        k_count = 0
        self.num_of_examples = 0
        for instance in self.instances:
            self.num_of_examples = self.num_of_examples + 1
            input_x = self.get_feat_vec(instance._feature_vector.features)
            assigned_k = k_count % self.K
            #print "Assigned K", assigned_k, "out of", self.K
            #self.cluster_matches[input_x] = assigned_k
            self.cluster_matches.append((input_x,assigned_k))
            k_count += 1


    def m_step(self):
        total = self.num_of_examples + self.K
        for k in range(self.K):
            arr = []
            #collect the xi's in the same cluster
            for input_x,cluster in self.cluster_matches:
                if cluster == k:
                    arr.append(input_x)

            #now, we get our mu_k
            mu = self.get_mu(arr)
            #and sigma_k
            sigma = self.get_sigma(mu,arr)
            #and phi_k
            phi = (len(arr) + 1)/ float(total)
            self.cluster_info[k] = (mu,sigma,phi)


    def get_mu(self, arr):
        #sum = [0 for i in range(len(arr[0]))]
        sum = 0
        #print "Get_mu arr:", arr
        for i in arr:
            sum = np.add(sum,i)
        mu = sum / float(len(arr))
        return mu

    def get_Sj(self,j):
        list1 = []
        for i,k in self.cluster_matches:
            list1.append(i[j])
        vari = 0.01*np.var(list1)
        #round(vari,1)
        return vari



    def get_sigma(self,mu,arr):
        sigma = []
        size = len(arr) - 1
        for j in range(len(mu)):
            sum = 0
            #list1 = []
            for i in range(len(arr)):
                add = abs((arr[i][j] - mu[j]))**2
                sum = sum + add
                #list1.append(arr[i][j])

            #MAYBE: try taking sqrt of sum
            sigma_j = sum / float(size)
            #if not working then make a method to get variance
            S_j = self.get_Sj(j)

            if sigma_j < S_j: sigma_j = S_j
            if len(arr) == 0 or len(arr) == 1: sigma_j = S_j
            sigma.append(sigma_j)
            #print "sigma_j", sigma_j, "and S_j", S_j
        #print "Sigma", sigma
        return sigma


    def log_likelihood(self, input_x, mu, sig):
        sum = 0
        for j in range(len(input_x)):
            if sig[j] == 0: continue
            first = 1 / math.sqrt((2*math.pi*sig[j]))
            exp = ((input_x[j] - mu[j])**2)/(2*sig[j])
            #exp = ((input_x[j] - mu[j]) / (2*sig[j]))**2
            second = math.exp(-exp)
            prod = first*second
            if prod == 0.0: continue
            else: sum = sum + math.log(prod,2)
            """
            try:
                sum = sum + math.log(prod,2)
            except ValueError:
                print input_x
                print ""
                print "Mu", mu
                print "exp", exp
                print "math_exp", np.exp(-1000)
                print "the j = ", j
                print "xi(j)", input_x[j]
                print "mu(j)", mu[j]
                print "sig(j)", sig[j]
                print "Product", prod
                print "Sum", sum
                quit()
            """

        return sum

    def train(self, instances):
        #print "Training.."
        self.assign_clusters()
        self.m_step()
        for instance in self.instances:
            input_x = self.get_feat_vec(instance._feature_vector.features)
            score_dict = {}
            for k in range(self.K):
                #print "Cluster_Info", self.cluster_info
                mu = self.cluster_info[k][0]
                sig = self.cluster_info[k][1]
                phi = self.cluster_info[k][2]
                sum = self.log_likelihood(input_x,mu,sig)
                score = sum*phi
                score_dict[score] = k
            #gives just the keyes
            sort_scores = sorted(score_dict, reverse=True)
            cluster = score_dict[sort_scores[0]]

            #now, we have new cluster
            our_sample = [cl for cl in self.cluster_matches if cl[0] == input_x][0]
            idx = self.cluster_matches.index(our_sample)
            #change cluster
            self.cluster_matches[idx] = (our_sample[0], cluster)
            self.m_step()


    def predict(self, instance):
        #print "Predicting..."
        input_x = self.get_feat_vec(instance._feature_vector.features)
        score_dict = {}
        for k in range(self.K):
            mu = self.cluster_info[k][0]
            sig = self.cluster_info[k][1]
            phi = self.cluster_info[k][2]
            sum = self.log_likelihood(input_x,mu,sig)
            score = sum*phi
            score_dict[score] = k

        sort_scores = sorted(score_dict, reverse=True)
        cluster = score_dict[sort_scores[0]]
        return cluster


    def get_feat_vec(self,feat_vec):
        x = [ 0 for j in range(self.high_idx)]
        for i in feat_vec: x[i-1] = feat_vec[i]
        return x
