package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

public class CoTrainerEnsemble extends CoTrainerAbstract {
    private Properties properties;

    @Override
    public Dataset Train_Classifiers(HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples,
                                     int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration, String original_arff_file,
                                     int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer, int exp_id, String arff,
                                     int iteration, double weight_for_log, boolean use_active_learning, int random_seed) throws Exception {

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        /* We start by partitioning the dataset based on the sets of features this function receives as a parameter */
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = dataset.replicateDatasetByColumnIndices(feature_sets.get(index));
            datasetPartitions.put(index, partition);
        }

        /* Randomly select the labeled instances from the training set. The remaining ones will be used as the unlabeled.
         * It is important that we use a fixed random seed for repeatability */
        List<Integer> labeledTrainingInstancesIndices = new ArrayList<>();
        Random rnd = new Random(random_seed);
        Fold trainingFold = dataset.getTrainingFolds().get(0); //there should be only ONE train fold
        while (labeledTrainingInstancesIndices.size() < initial_number_of_labled_samples) {
            int instanceIndex = rnd.nextInt(trainingFold.getNumOfInstancesInFold());
            if (!labeledTrainingInstancesIndices.contains(trainingFold.getIndices().get(instanceIndex))) {
                labeledTrainingInstancesIndices.add(trainingFold.getIndices().get(instanceIndex));
            }
        }

        /* If the unlabeled training set is larger than the specified parameter, we will sample X instances to
         * serve as the pool. TODO: replenish the pool upon sampling (although given the sizes it's not such a big deal */
        List<Integer> unlabeledSetIndices = new ArrayList<>();
        if (trainingFold.getIndices().size()-initial_number_of_labled_samples > initial_unlabeled_set_size) {
            //ToDo: add a random sampling function
        }
        else {
            for (int index : trainingFold.getIndices()) {
                if (!labeledTrainingInstancesIndices.contains(index)) {
                    unlabeledSetIndices.add(index);
                }
            }
        }

        List<Integer> labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);

        //And now we can begin the iterative process
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();
        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/

            //step 1 - get the list of unlabeled training samples
            List<Integer> unlabeledTrainingIndices = dataset.getIndicesOfTrainingInstances().stream().filter(p -> !labeledTrainingSetIndices.contains(p)).collect(Collectors.toList());

            //step 2 - train the classifiers on the labeled training set and run on the unlabeled training set
            for (int partitionIndex : feature_sets.keySet()) {
                EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingIndices),
                        new ArrayList<>(unlabeledTrainingIndices), properties);

                if (!evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                    evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
                }
                evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, i);
            }

            //step 3 - get the indices of the items we want to label (separately for each class)

        }





        //Dataset labeled_training_data, Dataset unlabeled_training_data, Dataset validation_set_data, Dataset test_data,
        return null;
    }

    @Override
    public void Previous_Iterations_Analysis(EvaluationPerIteraion models, Dataset training_set_data, Dataset validation_set_data, int current_iteration) {

    }

    @Override
    public String toString() {
        return null;
    }
}
