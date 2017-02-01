import os
import argparse
import sys
import pickle

from classes import ClassificationLabel, FeatureVector, Instance, Predictor
from perceptron import Perceptron, AveragePerceptron, PerceptronMargin
from pegasos import Pegasos
from knn import KNN,Distance_KNN
from adaboost import Adaboost
#from lambda_means import Lambda_Means
from lambda_means2 import Lambda_Means2
from naive_bayes import Naive_Bayes
from mc_perceptron import MC_Perceptron



def load_data(filename):
    instances = []
    #added
    highest_idx = 0
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()

            
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                    #added
                    if (index > highest_idx): highest_idx = index
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)
    #added
    return instances, highest_idx


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # TODO This is where you will add new command line options
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron",
                    default=1.0)

    parser.add_argument("--online-training-iterations", type=int,
                    help="The number of training iterations for online methods.", default=5)

    #adding lambda

    parser.add_argument("--pegasos-lambda", type=float, help="The regularization parameter for Pegasos.",
                    default=1e-4)

    #adding k value for k-nearest neighbors
    parser.add_argument("--knn", type=int, help="The value of K for KNN classification.",
                    default=5)

    parser.add_argument("--num-boosting-iterations", type=int, help="The number of boosting iterations to run.",
                    default=10)

    parser.add_argument("--num-clusters", type=int, help="The number of clusters in Naive Bayes clustering.",
                        default=3)

    parser.add_argument("--clustering-training-iterations", type=int,
                        help="The number of clustering iterations.", default=10)

    parser.add_argument("--cluster-lambda", type=float, help="The value of lambda in lambda-means.",
                    default=0.0)



    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(instances, algorithm, high_idx, learn_rate, iterate, peg_lambda, k_val, T, clus_lambda, K, clus_iter):

    if(algorithm == "perceptron"):
        classifier = Perceptron(instances, high_idx,learn_rate)
        #iterate the training
        for i in range(iterate):
            classifier.train(instances)
        return classifier

    elif(algorithm == "averaged_perceptron"):
        classifier2 = AveragePerceptron(instances, high_idx, learn_rate)
        for i in range(iterate):
            classifier2.train(instances)
        return classifier2

    elif(algorithm == "pegasos"):
        classifier3 = Pegasos(instances,high_idx,peg_lambda)
        for i in range(iterate):
            classifier3.train(instances)
        return classifier3

    elif(algorithm == "margin_perceptron"):
        classifier4 = PerceptronMargin(instances, high_idx, learn_rate, iterate)
        for i in range(iterate):
            classifier4.train(instances)
        return classifier4

    elif(algorithm == "knn"):
        classifier5 = KNN(instances,k_val, high_idx)
        for i in range(iterate):
            classifier5.train(instances)
        return classifier5

    elif(algorithm == "distance_knn"):
        classifier6 = Distance_KNN(instances,k_val, high_idx)
        for i in range(iterate):
            classifier6.train(instances)
        return classifier6

    elif(algorithm == "adaboost"):
        classifier7 = Adaboost(instances,T,high_idx)
        for i in range(iterate):
            classifier7.train(instances)
        return classifier7
    elif(algorithm == "lambda_means"):
        classifier8 = Lambda_Means2(instances, high_idx, clus_lambda, clus_iter)
        for i in range(iterate):
            #print "##################Training", i+1, "out of", iterate,"###############"
            classifier8.train(instances)
        return classifier8
    elif(algorithm == "nb_clustering"):
        classifier9 = Naive_Bayes(instances, high_idx, K)
        for i in range(iterate):
            #print "##################Training", i+1, "out of", iterate,"###############"
            classifier9.train(instances)
        return classifier9
    elif(algorithm == "mc_perceptron"):
        classifier10 = MC_Perceptron(instances,high_idx)
        for i in range(iterate):
            classifier10.train(instances)
        return classifier10


    else:
        return None


def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write('\n')

    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        #added
        instances, high_idx = load_data(args.data)
        learn_rate = args.online_learning_rate
        iterations = args.online_training_iterations
        peg_lambda = args.pegasos_lambda
        k_val = args.knn
        T_iter = args.num_boosting_iterations
        cluster_lam = args.cluster_lambda
        clus_iter = args.clustering_training_iterations
        num_clus = args.num_clusters


        # Train the model.
        predictor = train(instances, args.algorithm, high_idx, learn_rate, iterations, peg_lambda,k_val,T_iter, cluster_lam, num_clus,clus_iter)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        #added
        instances,__ = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

