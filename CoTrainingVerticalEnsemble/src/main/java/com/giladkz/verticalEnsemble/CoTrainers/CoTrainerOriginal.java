package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class CoTrainerOriginal extends CoTrainerAbstract {
    private Properties properties;

    @Override
    public Dataset Train_Classifiers(HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples,
                                     int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration, String original_arff_file,
                                     int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer, int exp_id, String arff,
                                     int iteration, double weight_for_log, boolean use_active_learning, int random_seed, List<Integer> labeledTrainingSet) throws Exception {

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
        List<Integer> labeledTrainingSetIndices;
        if (labeledTrainingSet.size() > 0 ){
            labeledTrainingSetIndices = new ArrayList<>(labeledTrainingSet);
        }else{
            labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);
        }


        List<Integer> unlabeledTrainingSetIndices = new ArrayList<>();
        Fold trainingFold = dataset.getTrainingFolds().get(0); //There should only be one training fold in this type of project
        if (trainingFold.getIndices().size()-initial_number_of_labled_samples > initial_unlabeled_set_size) {
            for (int index : trainingFold.getIndices()) {
                if (!labeledTrainingSetIndices.contains(index) && unlabeledTrainingSetIndices.size() < initial_unlabeled_set_size
                        && new Random().nextInt(100)< 96) {
                    unlabeledTrainingSetIndices.add(index);
                }
            }
        }
        else {
            for (int index : trainingFold.getIndices()) {
                if (!labeledTrainingSetIndices.contains(index)) {
                    unlabeledTrainingSetIndices.add(index);
                }
            }
        }

        //before we begin the co-training process, we test the performance on the original dataset
        System.out.println("Pre-run Original model results for dataset: "+original_arff_file+ " : ");
        RunExperimentsOnTestSet(exp_id, iteration, -1, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);

        //And now we can begin the iterative process
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();
        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/


            //step 1 - train the classifiers on the labeled training set and run on the unlabeled training set
            System.out.println("labaled: " + labeledTrainingSetIndices.size() + ";  unlabeled: " + unlabeledTrainingSetIndices.size() );

            Date expDate = new Date();
            Loader loader = new Loader();



            for (int partitionIndex : feature_sets.keySet()) {
                EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices),
                        new ArrayList<>(unlabeledTrainingSetIndices), properties);

                if (!evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                    evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
                }
                evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, i);

                //write unlabeled set
                /*
                for(int type=0; type < 2; type++){
                    if (type==0){
                        String tempFilePath = properties.getProperty("tempDirectory")+dataset.getName()+"_partition_"+partitionIndex+"_iteration_"+i+"_unlabeled_original_co_train.arff";
                        Files.deleteIfExists(Paths.get(tempFilePath));
                        ArffSaver s= new ArffSaver();
                        s.setInstances(datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices));
                        s.setFile(new File(tempFilePath));
                        s.writeBatch();
                    }
                    else {
                        String tempFilePath = properties.getProperty("tempDirectory")+dataset.getName()+"_partition_"+partitionIndex+"_iteration_"+i+"_labeled_original_co_train.arff";
                        Files.deleteIfExists(Paths.get(tempFilePath));
                        ArffSaver s= new ArffSaver();
                        s.setInstances(datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices));
                        s.setFile(new File(tempFilePath));
                        s.writeBatch();
                    }
                }*/
            }
            //Files.write( Paths.get(properties.getProperty("tempDirectory")+dataset.getName()+"_iteration_"+i+".txt")
            //        , ()->feature_sets.entrySet().stream().<CharSequence>map(e->e.getKey() + "," + e.getValue()).iterator());

            //step 2 - get the indices of the items we want to label (separately for each class)
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = new HashMap<>();
            HashMap<Integer, List<Integer>> instancesPerPartition = new HashMap<>();
            //these are the indices of the array provided to Weka. They need to be converted to the Dataset indices
            GetIndicesOfInstancesToLabelBasic(dataset, instances_per_class_per_iteration, evaluationResultsPerSetAndInteration, instancesToAddPerClass, random_seed, unlabeledTrainingSetIndices, instancesPerPartition);

            super.WriteInformationOnAddedItems(instancesToAddPerClass, i, exp_id,iteration,weight_for_log,instancesPerPartition, properties, dataset);

            //step 3 - set the class labels of the newly labeled instances to what we THINK they are
            for (int classIndex : instancesToAddPerClass.keySet()) {
                //because the columns of the partitions are actually the columns of the original dataset, there is no problem changing things only there
                dataset.updateInstanceTargetClassValue(new ArrayList<>(instancesToAddPerClass.get(classIndex).keySet()), classIndex);
            }


            //step 4 - add the selected instances to the labeled training set and remove them from the unlabeled set
            //IMPORTANT: when adding the unlabeled samples, it must be with the label I ASSUME they possess.
            List<Integer> allIndeicesToAdd = new ArrayList<>();
            for (int classIndex : instancesToAddPerClass.keySet()) {
                allIndeicesToAdd.addAll(new ArrayList<Integer>(instancesToAddPerClass.get(classIndex).keySet()));
            }
            labeledTrainingSetIndices.addAll(allIndeicesToAdd);
            unlabeledTrainingSetIndices = unlabeledTrainingSetIndices.stream().filter(line -> !allIndeicesToAdd.contains(line)).collect(Collectors.toList());

            //step 5 - train the models using the current instances and apply them to the test set
            System.out.println("Original model results for dataset: "+original_arff_file+" scores of iteration: " + i + ": ");
            RunExperimentsOnTestSet(exp_id, iteration, i, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);
        }
        return null;
    }

    @Override
    public String toString() {
        return "CoTrainerOriginal";
    }
}