# 1. enter selected algorithm here:
algorithm=(mc_perceptron)

# 2. enter the dataset options here:
options=(easy hard finance speech vision)
#### NOTE: the dataset: speech_mc is to be used ONLY with mc_perceptron, knn, distance_knn, and lambda_means


#3. For testing cluster algorithms (mc_perceptron, knn, distance_knn, and lambda_means): replace compute_accuracy.py with cluster_accuracy.py

for opt in "${options[@]}"; do
for algo in "${algorithm[@]}"; do
start=`date +%s`
if [ "$opt" == "easy" ] || [ "$opt" == "hard" ];then
python algorithms/classify.py --mode train --algorithm $algo --model-file synthetic/${opt}.$algo.model --data synthetic/${opt}.train

python algorithms/classify.py --mode test --model-file synthetic/${opt}.$algo.model --data synthetic/${opt}.dev --predictions-file synthetic/${opt}.dev.predictions
acc="$(python compute_accuracy.py synthetic/${opt}.dev synthetic/${opt}.dev.predictions)"
#acc2="$(python number_clusters.py synthetic/${opt}.dev)"

end=`date +%s`
echo "${opt} | $algo | $acc | $((end-start)) seconds"
#echo "$acc2"

elif [ "$opt" == "speech.mc" ];then
python algorithms/classify.py --mode train --algorithm $algo --model-file speech_hw5/${opt}.$algo.model --data speech_hw5/${opt}.train

python algorithms/classify.py --mode test --model-file speech_hw5/${opt}.$algo.model --data speech_hw5/${opt}.dev --predictions-file speech_hw5/${opt}.dev.predictions

acc="$(python compute_accuracy.py speech_hw5/${opt}.dev speech_hw5/${opt}.dev.predictions)"
end=`date +%s`
echo "${opt} | $algo | $acc | $((end-start)) seconds"

else
python algorithms/classify.py --mode train --algorithm $algo --model-file ${opt}/${opt}.$algo.model --data ${opt}/${opt}.train

python algorithms/classify.py --mode test --model-file ${opt}/${opt}.$algo.model --data ${opt}/${opt}.dev --predictions-file ${opt}/${opt}.dev.predictions


acc="$(python compute_accuracy.py ${opt}/${opt}.dev ${opt}/${opt}.dev.predictions)"
#acc2="$(python number_clusters.py ${opt}/${opt}.dev)"
end=`date +%s`
echo "${opt} | $algo | $acc | $((end-start)) seconds"
#echo "$acc2"

fi

done
done
