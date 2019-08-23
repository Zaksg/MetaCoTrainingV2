package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.MetaLearning.InstanceAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.InstancesBatchAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributesTdBatch;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.*;
import java.util.stream.Collectors;

public class CoTrainingMetaLearning extends CoTrainerAbstract {
    private Properties properties;

    @Override
    public Dataset Train_Classifiers(HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples,
                                     int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration, String original_arff_file,
                                     int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer, int exp_id, String arff,
                                     int iteration, double weight_for_log, boolean use_active_learning, int random_seed) throws Exception {

        /*This set is meta features analyzes the scores assigned to the unlabeled training set at each iteration.
        * Its possible uses include:
        * a) Providing a stopping criteria (i.e. need to go one iteration (or more) back because we're going off-course
        * b) Assisting in the selection of unlabeled samples to be added to the labeled set*/
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();
        InstanceAttributes instanceAttributes = new InstanceAttributes();
        InstancesBatchAttributes instancesBatchAttributes = new InstancesBatchAttributes();

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        //Data writing system
        //can be "sql" or "csv"
        String writeType = "csv";

        /* We start by partitioning the dataset based on the sets of features this function receives as a parameter */
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = dataset.replicateDatasetByColumnIndices(feature_sets.get(index));

            datasetPartitions.put(index, partition);
        }

        /* Randomly select the labeled instances from the training set. The remaining ones will be used as the unlabeled.
         * It is important that we use a fixed random seed for repeatability */
        List<Integer> labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);

        /* If the unlabeled training set is larger than the specified parameter, we will sample X instances to
         * serve as the pool. TODO: replenish the pool upon sampling (although given the sizes it's not such a big deal */
        List<Integer> unlabeledTrainingSetIndices = new ArrayList<>();
        Fold trainingFold = dataset.getTrainingFolds().get(0); //There should only be one training fold in this type of project
        if (trainingFold.getIndices().size()-initial_number_of_labled_samples > initial_unlabeled_set_size) {
            //ToDo: add a random sampling function
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
        RunExperimentsOnTestSet(exp_id, iteration, -1, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);

        //And now we can begin the iterative process

        //this object saves the results for the partitioned dataset. It is of the form parition -> iteration index -> results
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();

        //this object save the results of the runs of the unified datasets (original labeled + labeled during the co-training process).
        EvaluationPerIteraion unifiedDatasetEvaulationResults = new EvaluationPerIteraion();
        //write meta data information in groups and not one by one
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroup = new HashMap<>();
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeBatchMetaDataInGroup = new HashMap<>();
        HashMap<ArrayList<Integer>,int[]> writeInsertBatchInGroup = new HashMap<>();
        HashMap<int[], Double> writeSampleBatchScoreInGroup = new HashMap<>();
        int writeCounterBin = 1;

        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/


            //step 1 - train the classifiers on the labeled training set and run on the unlabeled training set
//            System.out.println("start iteration with: labaled: " + labeledTrainingSetIndices.size() + ";  unlabeled: " + unlabeledTrainingSetIndices.size() );

            for (int partitionIndex : feature_sets.keySet()) {
                EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices),
                        new ArrayList<>(unlabeledTrainingSetIndices), properties);

                if (!evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                    evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
                }
                evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, i);
            }

            //now we run the classifier trained on the unified set
            EvaluationInfo unifiedSetEvaluationResults = runClassifier(properties.getProperty("classifier"),
                    dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                    dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices),
                    new ArrayList<>(unlabeledTrainingSetIndices), properties);
            unifiedDatasetEvaulationResults.addEvaluationInfo(unifiedSetEvaluationResults, i);

            /*enter the meta features generation here*/
            Dataset labeledToMetaFeatures = dataset;
            Dataset unlabeledToMetaFeatures = dataset;
            int targetClassIndex = dataset.getMinorityClassIndex();
            boolean getDatasetInstancesSucc = false;
            for (int numberOfTries = 0; numberOfTries < 5 && !getDatasetInstancesSucc; numberOfTries++) {
                try{
                    labeledToMetaFeatures = getDataSetByInstancesIndices(dataset,labeledTrainingSetIndices,exp_id, -2, properties);
                    unlabeledToMetaFeatures = getDataSetByInstancesIndices(dataset,unlabeledTrainingSetIndices,exp_id, -2, properties);
                    getDatasetInstancesSucc = true;
                }catch (Exception e){
                    Thread.sleep(1000);
                    System.out.println("failed reading file, sleep for 1 second, for try: " + num_of_iterations);
                    getDatasetInstancesSucc = false;
                }
            }

            TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree = new TreeMap<>(evaluationResultsPerSetAndInteration);

            //score distribution
            TreeMap<Integer,AttributeInfo> scoreDistributionCurrentIteration = new TreeMap<>();
            scoreDistributionCurrentIteration = scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree
                    , unifiedDatasetEvaulationResults,
                    targetClassIndex/*dataset.getTargetColumnIndex()*/
                    ,"reg", properties);
            writeResultsToScoreDistribution(scoreDistributionCurrentIteration, i, exp_id, iteration, properties, dataset, writeType);

            ArrayList<ArrayList<Integer>> batchesInstancesList = new ArrayList<>();
            List<TreeMap<Integer,AttributeInfo>> instanceAttributeCurrentIterationList = new ArrayList<>();

            //batches generation
            Random rnd = new Random((i + Integer.parseInt(properties.getProperty("randomSeed"))));
            System.out.println("Started generating batches");
            //smart selection: get batches from the top 15% of classifiers scores
            if (Objects.equals(properties.getProperty("batchSelection"), "smart")){
                //create: structure: relative index -> [instance_pos, label]
                long batch_generation_start = System.currentTimeMillis();
                ArrayList<ArrayList<ArrayList<Integer>>> topSelectedInstancesCandidatesArr = getTopCandidates(evaluationResultsPerSetAndInteration, unlabeledTrainingSetIndices);
                long batch_generation_finish = System.currentTimeMillis();
                //System.out.println("Batch generation in " + (batch_generation_finish - batch_generation_start) + " ms");
                //generate batches
                int batchIndex = 0;
                HashMap<Character,int[]> pairsDict = new HashMap<>();
                pairsDict.put('a',new int[]{0,1});
                pairsDict.put('b',new int[]{0,2});
                pairsDict.put('c',new int[]{0,3});
                pairsDict.put('d',new int[]{1,2});
                pairsDict.put('e',new int[]{1,3});
                pairsDict.put('f',new int[]{2,3});
                for(char pair_0_0 : "abcdef".toCharArray()) {
                    for(char pair_0_1 : "abcdef".toCharArray()) {
                        for(char pair_1_0 : "abcdef".toCharArray()) {
                            for(char pair_1_1 : "abcdef".toCharArray()) {
                                //structure:
                                // [0]instancesBatchOrginalPos, [1]instancesBatchSelectedPos
                                // [2]assignedLabelsOriginalIndex_0, [3]assignedLabelsOriginalIndex_1
                                // [4]assignedLabelsSelectedIndex_0, [5]assignedLabelsSelectedIndex_1
                                long get_batch_start = System.currentTimeMillis();
                                ArrayList<ArrayList<Integer>> generatedBatch = new ArrayList<>();
                                generatedBatch = generateBatch(topSelectedInstancesCandidatesArr, pairsDict
                                        ,pair_0_0, pair_0_1, pair_1_0, pair_1_1);
                                HashMap<Integer, Integer> assignedLabelsOriginalIndex = new HashMap<>();
                                HashMap<Integer, Integer> assignedLabelsSelectedIndex = new HashMap<>();
                                long get_batch_finish = System.currentTimeMillis();
                                //System.out.println("get batch: " + batchIndex+" in " + (get_batch_finish - get_batch_start) + " ms");
                                //instance meta features
                                long instances_start = System.currentTimeMillis();
                                for (int instance_id=0; instance_id < generatedBatch.get(0).size(); instance_id++){
                                    int instancePos= generatedBatch.get(0).get(instance_id);
                                    int relativeIndex = generatedBatch.get(1).get(instance_id);
                                    int assignedClass=0;
                                    if (generatedBatch.get(3).contains(instancePos)){
                                        assignedClass=1;
                                    }
                                    assignedLabelsOriginalIndex.put(instancePos, assignedClass);
                                    assignedLabelsSelectedIndex.put(relativeIndex, assignedClass);
                                    //get instance meta features
                                    TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                                            unlabeledToMetaFeatures,dataset,
                                            i, evaluationResultsPerSetAndInterationTree,
                                            unifiedDatasetEvaulationResults, targetClassIndex,
                                            instancePos, assignedClass, properties);
                                    instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                                    int[] instanceInfoToWrite = new int[5];
                                    instanceInfoToWrite[0]=exp_id;
                                    instanceInfoToWrite[1]=iteration;
                                    instanceInfoToWrite[2]=i;
                                    instanceInfoToWrite[3]=instancePos;
                                    instanceInfoToWrite[4]=batchIndex;
                                    writeInstanceMetaDataInGroup.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
                                }
                                long instances_finish = System.currentTimeMillis();
                                //System.out.println("instances meta features: " + batchIndex+" in " + (instances_finish - instances_start) + " ms");
                                //batch meta features
                                //ToDo: check if these 2 rows needed
                                //batchesInstancesList.add(generatedBatch.get(0));
                                //batchInstancePosClass.put(batchIndex, new HashMap<>(assignedLabelsOriginalIndex));
                                long batch_start = System.currentTimeMillis();
                                TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                                        unlabeledToMetaFeatures,labeledToMetaFeatures,
                                        i, evaluationResultsPerSetAndInterationTree,
                                        unifiedDatasetEvaulationResults, targetClassIndex
                                        , generatedBatch.get(0) /*generatedBatch.get(1)*/
                                        , assignedLabelsOriginalIndex /*assignedLabelsSelectedIndex*/
                                        , properties);

                                int[] batchInfoToWrite = new int[3];
                                batchInfoToWrite[0]=exp_id;
                                batchInfoToWrite[1]=i;
                                batchInfoToWrite[2]=batchIndex;
                                writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);
                                batchIndex++;
                                long batch_finish = System.currentTimeMillis();
                                //System.out.println("batch meta features: " + batchIndex+" in " + (batch_finish - batch_start) + " ms");
                                //run the classifier with this batch: on cloned dataset and re-create the run-experiment method (Batches_Score)
                                long auc_start = System.currentTimeMillis();
                                /*
                                writeSampleBatchScoreInGroup.putAll(
                                        getBatchAucBeforeAndAfter(exp_id, iteration, true, i
                                                , batchIndex, dataset, dataset.getTestFolds().get(0)
                                                , dataset.getTrainingFolds().get(0)
                                                , datasetPartitions,assignedLabelsOriginalIndex, labeledTrainingSetIndices, properties));
                                */
                                long auc_finish = System.currentTimeMillis();
                                //System.out.println("auc batch: " + batchIndex+" in " + (auc_finish - auc_start) + " ms");
                                //calculate td-score-distribution
                                long td_start = System.currentTimeMillis();
                                TreeMap<Integer,AttributeInfo> tdScoreDistributionCurrentIteration = tdScoreDist(dataset, feature_sets
                                        , assignedLabelsOriginalIndex, labeledTrainingSetIndices, unlabeledTrainingSetIndices
                                        , evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults
                                        , dataset.getTestFolds().get(0), targetClassIndex, i, exp_id, batchIndex, properties);
                                //writeResultsToScoreDistribution(tdScoreDistributionCurrentIteration, i, exp_id, iteration, properties, dataset);
                                //ToDo: extract before and after auc to write to writeSampleBatchScoreInGroup and not to batch meta features!!
                                writeBatchMetaDataInGroup.put(tdScoreDistributionCurrentIteration, batchInfoToWrite);
                                long td_finish = System.currentTimeMillis();
                                //System.out.println("Td for batch: " + batchIndex+" in " + (td_finish - td_start) + " ms");
/*                                System.out.println("Total batch time: " + batchIndex+": total: " + (td_finish - get_batch_start) +
                                        " ms. instances meta features:" +(instances_finish - instances_start) +
                                        " ms. batch meta features: " + (batch_finish - batch_start) + " ms. auc: "+ (auc_finish - auc_start)
                                        + " ms. td: "+ (td_finish - td_start) + " ms.");*/
                            }
                            //scrumbel candidates
                            topSelectedInstancesCandidatesArr = getTopCandidates(evaluationResultsPerSetAndInteration, unlabeledTrainingSetIndices);
                        }
                        //scrumbel candidates
                        topSelectedInstancesCandidatesArr = getTopCandidates(evaluationResultsPerSetAndInteration, unlabeledTrainingSetIndices);
                    }
                }
                //end smart selection
            }
            //pick random 1000 batches of 8 instances and get meta features
            else{
                int numOfBatches = (int) Math.min(Integer.parseInt(properties.getProperty("numOfBatchedPerIteration")),Math.round(0.3*unlabeledTrainingSetIndices.size()));
                for (int batchIndex = 0; batchIndex < numOfBatches; batchIndex++) {
                    ArrayList<Integer> instancesBatchOrginalPos = new ArrayList<>();
                    ArrayList<Integer> instancesBatchSelectedPos = new ArrayList<>();

                    HashMap<Integer, Integer> assignedLabelsOriginalIndex = new HashMap<>();
                    HashMap<Integer, Integer> assignedLabelsSelectedIndex = new HashMap<>();
                    HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroupTemp = new HashMap<>();
                    int class0counter = 0;
                    int class1counter = 0;
                    for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
                        //we need 8 distinct instances
                        for (int sampleIndex = 0; sampleIndex < Integer.parseInt(properties.getProperty("instancesPerBatch"))/2; sampleIndex++) {
                            int relativeIndex = rnd.nextInt(unlabeledTrainingSetIndices.size());

                            while(assignedLabelsSelectedIndex.containsKey(relativeIndex)){
                                relativeIndex = rnd.nextInt(unlabeledTrainingSetIndices.size());
                            }
                            Integer instancePos = unlabeledTrainingSetIndices.get(relativeIndex);
                            //add instance pos to batch list
                            instancesBatchOrginalPos.add(instancePos);
                            instancesBatchSelectedPos.add(relativeIndex);

                            //calculate instance class

                            int assignedClass;
                            double scoreClass0 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().get(instancePos)[0];
                            double scoreClass1 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().get(instancePos)[1];
                            if (scoreClass0 > scoreClass1){
                                assignedClass = 0;
                                class0counter++;
                            }
                            else{
                                assignedClass = 1;
                                class1counter++;
                            }
                            assignedLabelsOriginalIndex.put(instancePos, assignedClass);
                            assignedLabelsSelectedIndex.put(relativeIndex, assignedClass);
                            //get instance meta features

                            TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                                    unlabeledToMetaFeatures,dataset,
                                    i, evaluationResultsPerSetAndInterationTree,
                                    unifiedDatasetEvaulationResults, targetClassIndex,
                                    instancePos, assignedClass, properties);
                            instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                            int[] instanceInfoToWrite = new int[5];
                            instanceInfoToWrite[0]=exp_id;
                            instanceInfoToWrite[1]=iteration;
                            instanceInfoToWrite[2]=i;
                            instanceInfoToWrite[3]=instancePos;
                            instanceInfoToWrite[4]=batchIndex;
                            writeInstanceMetaDataInGroupTemp.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
                            //writeResultsToInstanceMetaFeatures(instanceAttributeCurrentIteration, exp_id, iteration, i, instancePos, batchIndex, properties, dataset);
                        }
                    }
                    if (class0counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))
                            && class1counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))){
                        writeInstanceMetaDataInGroup.putAll(writeInstanceMetaDataInGroupTemp);
                        batchesInstancesList.add(instancesBatchOrginalPos);
                        TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                                unlabeledToMetaFeatures,labeledToMetaFeatures,
                                i, evaluationResultsPerSetAndInterationTree,
                                unifiedDatasetEvaulationResults
                                , targetClassIndex /*dataset.getTargetColumnIndex()*/
                                , instancesBatchOrginalPos /*instancesBatchSelectedPos*/
                                , assignedLabelsOriginalIndex /*assignedLabelsSelectedIndex*/
                                , properties);

                        int[] batchInfoToWrite = new int[3];
                        batchInfoToWrite[0]=exp_id;
                        batchInfoToWrite[1]=i;
                        batchInfoToWrite[2]=batchIndex;
                        writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);
                        //writeResultsToBatchesMetaFeatures(batchAttributeCurrentIterationList, exp_id, i, batchIndex, properties, dataset);

                        //run the classifier with this batch: on cloned dataset and re-create the run-experiment method (Batches_Score)
                        writeSampleBatchScoreInGroup.putAll(
                                getBatchAucBeforeAndAfter(exp_id, iteration, true, i
                                        , batchIndex, dataset, dataset.getTestFolds().get(0)
                                        , dataset.getTrainingFolds().get(0)
                                        , datasetPartitions,assignedLabelsOriginalIndex, labeledTrainingSetIndices, properties));
                            /*
                            runClassifierOnSampledBatch(exp_id, iteration, true, i
                                    , batchIndex, dataset, dataset.getTestFolds().get(0)
                                    , dataset.getTrainingFolds().get(0)
                                    , datasetPartitions,assignedLabelsOriginalIndex, labeledTrainingSetIndices, properties));
                            */
                        //calculate td-score-distribution
                        TreeMap<Integer,AttributeInfo> tdScoreDistributionCurrentIteration = tdScoreDist(dataset, feature_sets
                                , assignedLabelsOriginalIndex, labeledTrainingSetIndices, unlabeledTrainingSetIndices
                                , evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults
                                , dataset.getTestFolds().get(0), targetClassIndex, i, exp_id, batchIndex, properties);
                        //writeResultsToScoreDistribution(tdScoreDistributionCurrentIteration, i, exp_id, iteration, properties, dataset);
                        writeBatchMetaDataInGroup.put(tdScoreDistributionCurrentIteration, batchInfoToWrite);
                    }
                    writeInstanceMetaDataInGroupTemp.clear();
                }
            }
            System.out.println("finish insert all batches data");
            //step 2 - get the indices of the items we want to label (separately for each class)
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = new HashMap<>();
            HashMap<Integer, List<Integer>> instancesPerPartition = new HashMap<>();
            HashMap<Integer, Integer> selectedInstancesOrginalIndexes = new HashMap<>(); //index (relative) -> assigned class
            ArrayList<Integer> indicesOfAddedInstances = new ArrayList<>(); //index(original)
            //these are the indices of the array provided to Weka. They need to be converted to the Dataset indices
            GetIndicesOfInstancesToLabelBasicRelativeIndex(dataset, instances_per_class_per_iteration, evaluationResultsPerSetAndInteration, instancesToAddPerClass, random_seed, unlabeledTrainingSetIndices, instancesPerPartition, selectedInstancesOrginalIndexes, indicesOfAddedInstances);

            super.WriteInformationOnAddedItems(instancesToAddPerClass, i, exp_id,iteration,weight_for_log,instancesPerPartition, properties, dataset);

            //selected batch meta-data
            for (Integer instance: selectedInstancesOrginalIndexes.keySet()){
                //Integer originalInstancePos = unlabeledTrainingSetIndices.get(instance);
                Integer originalInstancePos = instance;
                Integer assignedClass = selectedInstancesOrginalIndexes.get(instance);
                TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                        unlabeledToMetaFeatures,dataset,
                        i, evaluationResultsPerSetAndInterationTree,
                        unifiedDatasetEvaulationResults, targetClassIndex/*dataset.getTargetColumnIndex()*/,
                        originalInstancePos, assignedClass, properties);
//                instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                int[] instanceInfoToWrite = new int[5];
                instanceInfoToWrite[0]=exp_id;
                instanceInfoToWrite[1]=iteration;
                instanceInfoToWrite[2]=i;
                instanceInfoToWrite[3]=originalInstancePos;
                instanceInfoToWrite[4]= -1;
                writeInstanceMetaDataInGroup.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
//                writeResultsToInstanceMetaFeatures(instanceAttributeCurrentIteration, exp_id, iteration, i, originalInstancePos, -1, properties, dataset);
            }

            TreeMap<Integer,AttributeInfo>  selectedBatchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree,
                    unifiedDatasetEvaulationResults, targetClassIndex/*dataset.getTargetColumnIndex()*/,
                    new ArrayList<>(selectedInstancesOrginalIndexes.keySet()), selectedInstancesOrginalIndexes, properties);
            int[] batchInfoToWrite = new int[3];
            batchInfoToWrite[0]=exp_id;
            batchInfoToWrite[1]=i;
            batchInfoToWrite[2]= -1 + iteration*(-1);
            writeBatchMetaDataInGroup.put(selectedBatchAttributeCurrentIterationList, batchInfoToWrite);
            //writeResultsToBatchesMetaFeatures(selectedBatchAttributeCurrentIterationList, exp_id, i,  -1 + iteration*(-1), properties, dataset);

            int[] insertBatchInfoToWrite = new int[3];
            insertBatchInfoToWrite[0]=exp_id;
            insertBatchInfoToWrite[1]=iteration;
            insertBatchInfoToWrite[2]=i;
            writeInsertBatchInGroup.put(indicesOfAddedInstances, insertBatchInfoToWrite);
            //writeToInstancesInBatchTbl(iteration*(-1), exp_id, iteration, indicesOfAddedInstances, properties);


            //write meta-data in groups of 20% of iteration
            if (i == (writeCounterBin*(num_of_iterations/5))-1){
                writeResultsToInstanceMetaFeaturesGroup(writeInstanceMetaDataInGroup, properties, dataset, exp_id, writeCounterBin, writeType);
                writeResultsToBatchMetaFeaturesGroup(writeBatchMetaDataInGroup, properties, dataset, exp_id, writeCounterBin, writeType);
                writeToInsertInstancesToBatchGroup(writeInsertBatchInGroup, properties, exp_id, writeCounterBin, writeType);
                writeToBatchScoreTblGroup(writeSampleBatchScoreInGroup, properties, exp_id, writeCounterBin, writeType);
                writeInstanceMetaDataInGroup.clear();
                writeBatchMetaDataInGroup.clear();
                writeInsertBatchInGroup.clear();
                writeSampleBatchScoreInGroup.clear();
                writeCounterBin++;
            }

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
            RunExperimentsOnTestSet(exp_id, iteration, i, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);

            System.out.println("dataset: "+dataset.getName()+" done insert batch and run the classifier for iteration: " + i);


            /* old version call
            step 6 - generate the meta features - not relevant!!
            generateMetaFeatures(dataset, labeledTrainingSetIndices, unlabeledTrainingSetIndices, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, i, properties);
            HashMap<Integer,AttributeInfo> scoreDistributionMetaFeatures = scoreDistributionBasedAttributes.getScoreDistributionBasedAttribute()            */
        }
        return null;
    }



    @Override
    public void Previous_Iterations_Analysis(EvaluationPerIteraion models, Dataset training_set_data, Dataset validation_set_data, int current_iteration) {

    }

    @Override
    public String toString() {
        return "CoTrainerMetaLearning";
    }


    /**
     * Used to generate the meta-features that will be used to guide the co-training process.
     * @param dataset The COMPLETE dataset (even the test set, so there's a need to be careful)
     * @param labeledTrainingSetIndices The indices of all the labeled training instances
     * @param unlabeledTrainingSetIndices The indices of all the unlabeled training instances
     * @param evaluationResultsPerSetAndInteration The evaluation results of EACH PARTITION across different indices
     * @param unifiedDatasetEvaulationResults The evaluation results of the UNIFIED (i.e. all features) features set across differnet iterations
     * @param currentIterationIndex The index of the current iteration
     * @param properties Configuration values
     * @return
     * @throws Exception
     */
    private HashMap<Integer,AttributeInfo> generateMetaFeatures(
            Dataset dataset, List<Integer> labeledTrainingSetIndices
            , List<Integer> unlabeledTrainingSetIndices
            , HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , EvaluationPerIteraion unifiedDatasetEvaulationResults
            , int currentIterationIndex, Properties properties) throws Exception{


        Loader loader = new Loader();
        String tempFilePath = properties.getProperty("tempDirectory") + "temp.arff";
        Files.deleteIfExists(Paths.get(tempFilePath));
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1,-1,0,0,0,-1,true, FoldsInfo.foldType.Train);

        //generate the labeled training instances dataset
        ArffSaver s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader = new BufferedReader(new FileReader(tempFilePath));
        Dataset labeledTrainingDataset = loader.readArff(reader, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader.close();

        File file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted1");
        }

        //generate the unlabeled training instances dataset
        s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader1 = new BufferedReader(new FileReader(tempFilePath));
        Dataset unlabeledTrainingDataset = loader.readArff(reader1, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader1.close();

        file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted2");
        }

        //evaluationResultsPerSetAndInteration
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();
        InstanceAttributes instanceAttributes = new InstanceAttributes();

        int targetClassIndex = dataset.getMinorityClassIndex();

        //scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes(labeledTrainingDataset,unlabeledTrainingDataset, currentIterationIndex, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, dataset.getTargetColumnIndex(), properties);

//        for (int instancePos: unlabeledTrainingDataset.getIndices()) {
//            instanceAttributes.getInstanceAssignmentMetaFeatures(labeledTrainingDataset, unlabeledTrainingDataset, currentIterationIndex, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, targetClassIndex, instancePos, unlabeledTrainingDataset.getInstancesClassByIndex(Arrays.asList(instancePos)).get(0), properties);
//        }
        return null;

    }

    private Dataset getDataSetByInstancesIndices(Dataset dataset, List<Integer> setIndices, int exp_id, int batchIndex, Properties properties)throws Exception{

        Date expDate = new Date();
        Loader loader = new Loader();
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1
                ,-1,0,0,0,-1
                ,true, FoldsInfo.foldType.Train);
        /*
        //generate dataset with files - generate the labeled training instances dataset
        String tempFilePath = properties.getProperty("tempDirectory")+exp_id+"_"+batchIndex+"_"+expDate.getTime()+"_temp.arff";
        Files.deleteIfExists(Paths.get(tempFilePath));
        ArffSaver s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,setIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader = new BufferedReader(new FileReader(tempFilePath));
        Dataset newDataset_file = loader.readArff(reader, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader.close();
        //need to delete temp files eventually
        File file = new File(tempFilePath);
        if(!file.delete())
        {
            throw new Exception("Temp file not deleted1");
        }*/
        Instances indicedInstances = dataset.generateSet(FoldsInfo.foldType.Train, setIndices);
        Dataset newDataset = loader.readArff(indicedInstances, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo
                , dataset, FoldsInfo.foldType.Train, setIndices);

        return newDataset;
    }

    private void writeToBatchScoreTblGroup(HashMap<int[], Double> writeSampleBatchScoreInGroup, Properties properties, int exp_id, int writeNum, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
            for (Map.Entry<int[], Double> outerEntry : writeSampleBatchScoreInGroup.entrySet()) {
                String sql = "insert into tbl_Batchs_Score(att_id, batch_id, exp_id, exp_iteration, score_type, score_value, test_set_size) values (?, ?, ?, ?, ?, ?, ?)";
                Double auc = outerEntry.getValue();
                int[] info = outerEntry.getKey();
                int att_id = 0;
                //insert to table
                String score_type;
                if (info[4] < 0) {
                    score_type = "auc_before_add_batch";
                } else {
                    score_type = "auc_after_add_batch";
                }
                PreparedStatement preparedStmt = conn.prepareStatement(sql);
                preparedStmt.setInt(1, att_id);
                preparedStmt.setInt(2, info[0]);
                preparedStmt.setInt(3, info[1]);
                preparedStmt.setInt(4, info[2]);
                preparedStmt.setString(5, score_type);
                preparedStmt.setDouble(6, auc);
                preparedStmt.setDouble(7, info[3]);

                preparedStmt.execute();
                preparedStmt.close();

                att_id++;

            }
            conn.close();
        }
        //csv
        else{
            String folderPath = properties.getProperty("modelFiles");
            String filename = "tbl_Batches_Score_exp_"+exp_id+"_writeNum_"+writeNum+".csv";
            FileWriter fileWriter = new FileWriter(folderPath+filename);
            String fileHeader = "att_id,batch_id,exp_id,exp_iteration,score_type,score_value,test_set_size\n";
            fileWriter.append(fileHeader);
            int att_id = 0;
            for (Map.Entry<int[], Double> outerEntry : writeSampleBatchScoreInGroup.entrySet()) {
                Double auc = outerEntry.getValue();
                int[] info = outerEntry.getKey();
                //int att_id = 0;
                //insert to table
                String score_type;
                if (info[4] < 0) {
                    score_type = "auc_before_add_batch";
                } else {
                    score_type = "auc_after_add_batch";
                }
                fileWriter.append(String.valueOf(att_id));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(info[0]));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(info[1]));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(info[2]));
                fileWriter.append(",");
                fileWriter.append(score_type);
                fileWriter.append(",");
                fileWriter.append(String.valueOf(auc));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(info[3]));
                fileWriter.append("\n");
                att_id++;
            }
            fileWriter.flush();
            fileWriter.close();
            System.out.println("Wrote this file: " + folderPath+filename);
        }
    }

    private void writeToBatchScoreTbl(int batch_id, int exp_id, int exp_iteration,
                                      String score_type, double score, int test_set_size, Properties properties)throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Batchs_Score(att_id, batch_id, exp_id, exp_iteration, score_type, score_value, test_set_size) values (?, ?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        int att_id=0;
        //insert to table
        PreparedStatement preparedStmt = conn.prepareStatement(sql);
        preparedStmt.setInt (1, att_id);
        preparedStmt.setInt (2, batch_id);
        preparedStmt.setInt (3, exp_id);
        preparedStmt.setInt (4, exp_iteration);
        preparedStmt.setString (5, score_type);
        preparedStmt.setDouble (6, score);
        preparedStmt.setDouble (7, test_set_size);

        preparedStmt.execute();
        preparedStmt.close();

        att_id++;
        conn.close();
    }

    private void writeResultsToScoreDistribution(TreeMap<Integer, AttributeInfo> scroeDistData, int expID, int expIteration,
                                                 int innerIteration, Properties properties, Dataset dataset, String writeType) throws Exception {

        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);

            String sql = "insert into tbl_Score_Distribution_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";

            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            int att_id = 0;
            for (Map.Entry<Integer, AttributeInfo> entry : scroeDistData.entrySet()) {
                String metaFeatureName = entry.getValue().getAttributeName();
                Object metaFeatureValueRaw = entry.getValue().getValue();

                //cast results to double
                Double metaFeatureValue = null;
                if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = (Double) metaFeatureValueRaw;
                } else if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = ((Double) metaFeatureValueRaw).doubleValue();
                }
                if (Double.isNaN(metaFeatureValue)) {
                    metaFeatureValue = -1.0;
                }

                //insert to table
                PreparedStatement preparedStmt = conn.prepareStatement(sql);
                preparedStmt.setInt(1, att_id);
                preparedStmt.setInt(2, expID);
                preparedStmt.setInt(3, expIteration);
                preparedStmt.setInt(4, innerIteration);
                preparedStmt.setString(5, metaFeatureName);
                preparedStmt.setDouble(6, metaFeatureValue);

                preparedStmt.execute();
                preparedStmt.close();

                att_id++;
            }
            conn.close();
        }
        //csv
        else{
            String folderPath = properties.getProperty("modelFiles");
            String filename = "tbl_Score_Distribution_Meta_Data_exp_"+expIteration+"_starting_iteration_"+innerIteration+expID+".csv";
            FileWriter fileWriter = new FileWriter(folderPath+filename);
            String fileHeader = "att_id,exp_id,exp_iteration,inner_iteration_id,meta_feature_name,meta_feature_value\n";
            fileWriter.append(fileHeader);
            int att_id = 0;
            for (Map.Entry<Integer, AttributeInfo> entry : scroeDistData.entrySet()) {
                String metaFeatureName = entry.getValue().getAttributeName();
                Object metaFeatureValueRaw = entry.getValue().getValue();

                //cast results to double
                Double metaFeatureValue = null;
                if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = (Double) metaFeatureValueRaw;
                } else if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = ((Double) metaFeatureValueRaw).doubleValue();
                }
                if (Double.isNaN(metaFeatureValue)) {
                    metaFeatureValue = -1.0;
                }
                fileWriter.append(String.valueOf(att_id));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(expID));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(expIteration));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(innerIteration));
                fileWriter.append(",");
                fileWriter.append(metaFeatureName);
                fileWriter.append(",");
                fileWriter.append(String.valueOf(metaFeatureValue));
                fileWriter.append("\n");
                att_id++;
            }
            fileWriter.flush();
            fileWriter.close();
            System.out.println("Wrote this file: " + folderPath+filename);
            /*
            //pivot table pilot
            String pivot_filename = "pivot_tbl_Score_Distribution_Meta_Data_exp_"+expIteration+"_starting_iteration_"+innerIteration+expID+".csv";
            FileWriter pivot_fileWriter = new FileWriter(folderPath+pivot_filename);
            String pivot_fileHeader = "att_id,exp_id,exp_iteration,inner_iteration_id";
            for (Map.Entry<Integer, AttributeInfo> entry : scroeDistData.entrySet()) {
                String metaFeatureName = entry.getValue().getAttributeName();
                pivot_fileHeader+=","+metaFeatureName;
            }
            pivot_fileWriter.append(pivot_fileHeader+"\n");

            pivot_fileWriter.append(String.valueOf(att_id));
            pivot_fileWriter.append(",");
            pivot_fileWriter.append(String.valueOf(expID));
            pivot_fileWriter.append(",");
            pivot_fileWriter.append(String.valueOf(expIteration));
            pivot_fileWriter.append(",");
            pivot_fileWriter.append(String.valueOf(innerIteration));

            for (Map.Entry<Integer, AttributeInfo> entry : scroeDistData.entrySet()) {
                pivot_fileWriter.append(",");
                Object metaFeatureValueRaw = entry.getValue().getValue();
                //cast results to double
                Double metaFeatureValue = null;
                if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = (Double) metaFeatureValueRaw;
                } else if (metaFeatureValueRaw instanceof Double) {
                    metaFeatureValue = ((Double) metaFeatureValueRaw).doubleValue();
                }
                if (Double.isNaN(metaFeatureValue)) {
                    metaFeatureValue = -1.0;
                }
                pivot_fileWriter.append(String.valueOf(metaFeatureValue));
            }
            pivot_fileWriter.flush();
            pivot_fileWriter.close();*/
        }
    }


    private void writeResultsToInstanceMetaFeaturesGroup(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup, Properties properties, Dataset dataset, int expId, int writeNum, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeInstanceMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";
                TreeMap<Integer, AttributeInfo> instanceMetaData = outerEntry.getKey();
                int[] instanceInfo = outerEntry.getValue();

                int att_id = 0;
                for (Map.Entry<Integer, AttributeInfo> entry : instanceMetaData.entrySet()) {
                    String metaFeatureName = entry.getValue().getAttributeName();
                    String metaFeatureValue = entry.getValue().getValue().toString();

                    try {
                        //insert to table
                        PreparedStatement preparedStmt = conn.prepareStatement(sql);
                        preparedStmt.setInt(1, att_id);
                        preparedStmt.setInt(2, instanceInfo[0]);
                        preparedStmt.setInt(3, instanceInfo[1]);
                        preparedStmt.setInt(4, instanceInfo[2]);
                        preparedStmt.setInt(5, instanceInfo[3]);
                        preparedStmt.setInt(6, instanceInfo[4]);
                        preparedStmt.setString(7, metaFeatureName);
                        preparedStmt.setString(8, metaFeatureValue);

                        preparedStmt.execute();
                        preparedStmt.close();

                        att_id++;
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.out.println("failed insert instance for: (" + att_id + ", " + instanceInfo[0] + ", " + instanceInfo[1] + ", " + instanceInfo[2]
                                + ", " + instanceInfo[3] + ", " + instanceInfo[4] + ", " + metaFeatureName + ", " + metaFeatureValue + ")");
                    }
                }
            }
            conn.close();
        }
        //csv
        else{
            String folderPath = properties.getProperty("modelFiles");
            String filename = "tbl_Instances_Meta_Data_exp_"+expId+"_writeNum_"+writeNum+".csv";
            FileWriter fileWriter = new FileWriter(folderPath+filename);
            String fileHeader = "att_id,exp_id,exp_iteration,inner_iteration_id,instance_pos,batch_id,meta_feature_name,meta_feature_value\n";
            fileWriter.append(fileHeader);
            Iterator<Map.Entry<TreeMap<Integer, AttributeInfo>, int[]>> itr = writeInstanceMetaDataInGroup.entrySet().iterator();
            //for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeInstanceMetaDataInGroup.entrySet()) {
            while(itr.hasNext()){
                Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry = itr.next();

                TreeMap<Integer, AttributeInfo> instanceMetaData = outerEntry.getKey();
                int[] instanceInfo = outerEntry.getValue();

                int att_id = 0;
                Iterator<Map.Entry<Integer, AttributeInfo>> itr2 = instanceMetaData.entrySet().iterator();
                //for (Map.Entry<Integer, AttributeInfo> entry : instanceMetaData.entrySet()) {
                while(itr2.hasNext()){
                    Map.Entry<Integer, AttributeInfo> entry = itr2.next();
                    String metaFeatureName = entry.getValue().getAttributeName();
                    String metaFeatureValue = entry.getValue().getValue().toString();

                    try {
                        //insert to table
                        fileWriter.append(String.valueOf(att_id));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceInfo[0]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceInfo[1]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceInfo[2]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceInfo[3]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceInfo[4]));
                        fileWriter.append(",");
                        fileWriter.append(metaFeatureName);
                        fileWriter.append(",");
                        fileWriter.append(metaFeatureValue);
                        fileWriter.append("\n");
                        att_id++;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            fileWriter.flush();
            fileWriter.close();
            System.out.println("Wrote this file: " + folderPath+filename);
            //return folderPath+filename;
        }
    }

    private void writeResultsToInstanceMetaFeatures(TreeMap<Integer,AttributeInfo> instanceMetaData, int expID, int expIteration,
                                                    int innerIteration, int instancePos, int batchId, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        int att_id = 0;
        for(Map.Entry<Integer,AttributeInfo> entry : instanceMetaData.entrySet()){
            String metaFeatureName = entry.getValue().getAttributeName();
            String metaFeatureValue = entry.getValue().getValue().toString();

            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, att_id);
            preparedStmt.setInt (2, expID);
            preparedStmt.setInt (3, expIteration);
            preparedStmt.setInt(4, innerIteration);
            preparedStmt.setInt(5, instancePos);
            preparedStmt.setInt (6, batchId);
            preparedStmt.setString   (7, metaFeatureName);
            preparedStmt.setString   (8, metaFeatureValue);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;
        }
        conn.close();
    }

    private void writeResultsToBatchMetaFeaturesGroup(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup, Properties properties, Dataset dataset, int exp_id, int writeNum, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeBatchMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";
                TreeMap<Integer, AttributeInfo> batchMetaData = outerEntry.getKey();
                int[] batchInfo = outerEntry.getValue();
                int att_id = 0;
                for (Map.Entry<Integer, AttributeInfo> entry : batchMetaData.entrySet()) {
                    String metaFeatureName = entry.getValue().getAttributeName();
                    String metaFeatureValue = entry.getValue().getValue().toString();
                    try {
                        //insert to table
                        PreparedStatement preparedStmt = conn.prepareStatement(sql);
                        preparedStmt.setInt(1, att_id);
                        preparedStmt.setInt(2, batchInfo[0]);
                        preparedStmt.setInt(3, batchInfo[1]);
                        preparedStmt.setInt(4, batchInfo[2]);
                        preparedStmt.setString(5, metaFeatureName);
                        preparedStmt.setString(6, metaFeatureValue);

                        preparedStmt.execute();
                        preparedStmt.close();

                        att_id++;
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.out.println("failed insert batch for: (" + att_id + ", " + batchInfo[0] + ", " + batchInfo[1] + ", " + batchInfo[2]
                                + ", " + metaFeatureName + ", " + metaFeatureValue + ")");
                    }

                }
            }
            conn.close();
        }
        //csv
        else{
            String folderPath = properties.getProperty("modelFiles");
            String filename = "tbl_Batches_Meta_Data_exp_"+exp_id+"_writeNum_"+writeNum+".csv";
            FileWriter fileWriter = new FileWriter(folderPath+filename);
            String fileHeader = "att_id,exp_id,exp_iteration,batch_id,meta_feature_name,meta_feature_value\n";
            fileWriter.append(fileHeader);
            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeBatchMetaDataInGroup.entrySet()) {
                TreeMap<Integer, AttributeInfo> batchMetaData = outerEntry.getKey();
                int[] batchInfo = outerEntry.getValue();
                int att_id = 0;
                for (Map.Entry<Integer, AttributeInfo> entry : batchMetaData.entrySet()) {
                    String metaFeatureName = entry.getValue().getAttributeName();
                    String metaFeatureValue = entry.getValue().getValue().toString();
                    try {
                        //insert to table
                        fileWriter.append(String.valueOf(att_id));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(batchInfo[0]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(batchInfo[1]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(batchInfo[2]));
                        fileWriter.append(",");
                        fileWriter.append(metaFeatureName);
                        fileWriter.append(",");
                        fileWriter.append(metaFeatureValue);
                        fileWriter.append("\n");
                        att_id++;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            fileWriter.flush();
            fileWriter.close();
            System.out.println("Wrote this file: " + folderPath+filename);
            //return folderPath+filename;
        }
    }

    private void writeResultsToBatchesMetaFeatures(TreeMap<Integer,AttributeInfo> batchMetaData, int expID, int innerIteration,
                                                   int batchID, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        int att_id = 0;
        for(Map.Entry<Integer,AttributeInfo> entry : batchMetaData.entrySet()){
            String metaFeatureName = entry.getValue().getAttributeName();
            String metaFeatureValue = entry.getValue().getValue().toString();

            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt(1, att_id);
            preparedStmt.setInt(2, expID);
            preparedStmt.setInt(3, innerIteration);
            preparedStmt.setInt(4, batchID);
            preparedStmt.setString(5, metaFeatureName);
            preparedStmt.setString(6, metaFeatureValue);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;
        }
        conn.close();
    }

    private void writeToInsertInstancesToBatchGroup(HashMap<ArrayList<Integer>, int[]> writeInsertBatchInGroup, Properties properties, int exp_id, int writeNum, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
            for (Map.Entry<ArrayList<Integer>, int[]> outerEntry : writeInsertBatchInGroup.entrySet()) {
                String sql = "insert into tbl_Instance_In_Batch(exp_id, exp_iteration, inner_iteration_id,batch_id, instance_pos) values (?, ?, ?, ?, ?)";
                ArrayList<Integer> instancesBatchPos = outerEntry.getKey();
                int[] instanceToBatch = outerEntry.getValue();
                for (Integer instancePosInBatch : instancesBatchPos) {
                    //insert to table
                    int batch_id = instanceToBatch[2] * (-1) - 1;
                    PreparedStatement preparedStmt = conn.prepareStatement(sql);
                    preparedStmt.setInt(1, instanceToBatch[0]);
                    preparedStmt.setInt(2, instanceToBatch[1]);
                    preparedStmt.setInt(3, instanceToBatch[2]);
                    preparedStmt.setInt(4, batch_id);
                    preparedStmt.setInt(5, instancePosInBatch);

                    preparedStmt.execute();
                    preparedStmt.close();
                }
            }
            conn.close();
        }
        //csv
        else{
            String folderPath = properties.getProperty("modelFiles");
            String filename = "tbl_Instance_In_Batch_exp_"+exp_id+"_writeNum_"+writeNum+".csv";
            FileWriter fileWriter = new FileWriter(folderPath+filename);
            String fileHeader = "exp_id,exp_iteration,inner_iteration_id,batch_id,instance_pos\n";
            fileWriter.append(fileHeader);
            for (Map.Entry<ArrayList<Integer>, int[]> outerEntry : writeInsertBatchInGroup.entrySet()) {
                ArrayList<Integer> instancesBatchPos = outerEntry.getKey();
                int[] instanceToBatch = outerEntry.getValue();
                int att_id = 0;
                for (Integer instancePosInBatch : instancesBatchPos) {
                    int batch_id = instanceToBatch[2] * (-1) - 1;
                    try {
                        //insert to table
                        fileWriter.append(String.valueOf(instanceToBatch[0]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceToBatch[1]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instanceToBatch[2]));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(batch_id));
                        fileWriter.append(",");
                        fileWriter.append(String.valueOf(instancePosInBatch));
                        fileWriter.append("\n");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            fileWriter.flush();
            fileWriter.close();
            System.out.println("Wrote this file: " + folderPath+filename);
        }
    }

    private void writeToInstancesInBatchTbl(int batch_id, int exp_id, int exp_iteration,
                                            ArrayList<Integer> instancesBatchPos, Properties properties)throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Instance_In_Batch(batch_id, exp_id, exp_iteration, instance_id, instance_pos) values (?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for(Integer instancePosInBatch : instancesBatchPos){
            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, batch_id);
            preparedStmt.setInt (2, exp_id);
            preparedStmt.setInt (3, exp_iteration);
            preparedStmt.setInt (4, instancePosInBatch);
            preparedStmt.setInt (5, instancePosInBatch);

            preparedStmt.execute();
            preparedStmt.close();
        }
        conn.close();
    }

    private Dataset generateDatasetCopyWithBatchAdded (
            Dataset dataset, HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {
        //clone the original dataset
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        //add batch instances to the cloned dataset
        ArrayList<Integer> instancesClass0 = new ArrayList<>();
        ArrayList<Integer> instancesClass1 = new ArrayList<>();
        for (Map.Entry<Integer,Integer> entry : batchInstancesToAdd.entrySet()){
            int instancePos = entry.getKey();
            int classIndex = entry.getValue();
            if (classIndex == 0){
                instancesClass0.add(instancePos);
            }
            else{
                instancesClass1.add(instancePos);
            }
            clonedlabeLedTrainingSetIndices.add(instancePos);
        }
        clonedDataset.updateInstanceTargetClassValue(instancesClass0, 0);
        clonedDataset.updateInstanceTargetClassValue(instancesClass1, 1);
        return clonedDataset;
    }

    private HashMap<int[], Double> getAucBeforeSampledBatch(
            int expID, int expIteration, Boolean isAUC
            , int innerIteration, int batch_id
            , Dataset dataset, Fold testFold, Fold trainFold
            , HashMap<Integer,Dataset> datasetPartitions
            , HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {
        HashMap<int[], Double> result = new HashMap<>();
        //clone the original dataset
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        //run AUC before add the batch
        AUC aucBeforeAddBatch = new AUC();
        int[] testFoldLabelsBeforeAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsBeforeAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                new ArrayList<>(testFold.getIndices()), properties);
        double measureAucBeforeAddBatch = aucBeforeAddBatch.measure
                (testFoldLabelsBeforeAdding
                        , getSingleClassValueConfidenceScore(evaluationResultsBeforeAdding.getScoreDistributions()
                        ,1));
        int[] infoListBefore = new int[5];
        infoListBefore[0] = batch_id;
        infoListBefore[1] = expID;
        infoListBefore[2] = innerIteration;
        infoListBefore[3] = testFoldLabelsBeforeAdding.length;
        infoListBefore[4] = -1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
        result.put(infoListBefore, measureAucBeforeAddBatch);
        //writeToBatchScoreTbl(batch_id,expID, innerIteration, "auc_before_add_batch", measureAucBeforeAddBatch, testFoldLabelsBeforeAdding.length, properties);
        return result;
    }

    private HashMap<int[], Double> getAucAfterSampledBatch(
            int expID, int expIteration, Boolean isAUC
            , int innerIteration, int batch_id
            , Dataset clonedDataset, Fold testFold, Fold trainFold
            , HashMap<Integer,Dataset> datasetPartitions
            , HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> clonedlabeLedTrainingSetIndices
            , Properties properties) throws Exception {

        HashMap<int[], Double> result = new HashMap<>();
        //run classifier after adding
        int[] testFoldLabelsAfterAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        //AUC after change dataset
        AUC aucAfterAddBatch = new AUC();
        double measureAucAfterAddBatch = aucAfterAddBatch.measure
                (testFoldLabelsAfterAdding,
                        getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions()
                                ,1));
        int[] infoListAfter = new int[5];
        infoListAfter[0] = batch_id;
        infoListAfter[1] = expID;
        infoListAfter[2] = innerIteration;
        infoListAfter[3] = testFoldLabelsAfterAdding.length;
        infoListAfter[4] = 1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
        result.put(infoListAfter, measureAucAfterAddBatch);
        //writeToBatchScoreTbl(batch_id, expID, innerIteration, "auc_after_add_batch", measureAucAfterAddBatch, testFoldLabelsAfterAdding.length, properties);
        return result;
    }

    private HashMap<int[], Double> getBatchAucBeforeAndAfter(
            int expID, int expIteration, Boolean isAUC
            , int innerIteration, int batch_id
            , Dataset dataset, Fold testFold, Fold trainFold
            , HashMap<Integer,Dataset> datasetPartitions
            , HashMap<Integer, Integer> batchInstancesToAdd
            , List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {
        HashMap<int[], Double> result = new HashMap<>();
        //AUC before
        result.putAll(
                getAucBeforeSampledBatch(expID, expIteration, isAUC, innerIteration, batch_id
                    , dataset, testFold, trainFold, datasetPartitions, batchInstancesToAdd
                        , labeledTrainingSetIndices, properties));
        //Add batch do dataset
        Dataset clonedDataset = generateDatasetCopyWithBatchAdded(dataset, batchInstancesToAdd
                , labeledTrainingSetIndices,  properties);
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        for (Map.Entry<Integer,Integer> entry : batchInstancesToAdd.entrySet()){
            int instancePos = entry.getKey();
            clonedlabeLedTrainingSetIndices.add(instancePos);
        }
        //AUC after
        result.putAll(
                getAucAfterSampledBatch(expID, expIteration, isAUC, innerIteration, batch_id
                        , clonedDataset, testFold, trainFold, datasetPartitions, batchInstancesToAdd
                        , clonedlabeLedTrainingSetIndices, properties));
        return result;
    }

    private HashMap<int[], Double> runClassifierOnSampledBatch(
            int expID, int expIteration, Boolean isAUC
            , int innerIteration, int batch_id
            , Dataset dataset, Fold testFold, Fold trainFold
            , HashMap<Integer,Dataset> datasetPartitions
            , HashMap<Integer, Integer> batchInstancesToAdd
            , List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {
        HashMap<int[], Double> result = new HashMap<>();
        //clone the original dataset
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        //run AUC before add the batch
        if (isAUC){
            AUC aucBeforeAddBatch = new AUC();
            int[] testFoldLabelsBeforeAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
            //Test the entire newly-labeled training set on the test set
            EvaluationInfo evaluationResultsBeforeAdding = runClassifier(properties.getProperty("classifier"),
                    clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                    clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                    new ArrayList<>(testFold.getIndices()), properties);
            double measureAucBeforeAddBatch = aucBeforeAddBatch.measure
                    (testFoldLabelsBeforeAdding
                            , getSingleClassValueConfidenceScore(evaluationResultsBeforeAdding.getScoreDistributions()
                            ,1));
            int[] infoListBefore = new int[5];
            infoListBefore[0] = batch_id;
            infoListBefore[1] = expID;
            infoListBefore[2] = innerIteration;
            infoListBefore[3] = testFoldLabelsBeforeAdding.length;
            infoListBefore[4] = -1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
            result.put(infoListBefore, measureAucBeforeAddBatch);
            //writeToBatchScoreTbl(batch_id,expID, innerIteration, "auc_before_add_batch", measureAucBeforeAddBatch, testFoldLabelsBeforeAdding.length, properties);
        }

        //add batch instances to the cloned dataset
        ArrayList<Integer> instancesClass0 = new ArrayList<>();
        ArrayList<Integer> instancesClass1 = new ArrayList<>();
        for (Map.Entry<Integer,Integer> entry : batchInstancesToAdd.entrySet()){
            int instancePos = entry.getKey();
            int classIndex = entry.getValue();
            if (classIndex == 0){
                instancesClass0.add(instancePos);
            }
            else{
                instancesClass1.add(instancePos);
            }
            clonedlabeLedTrainingSetIndices.add(instancePos);
        }
        clonedDataset.updateInstanceTargetClassValue(instancesClass0, 0);
        clonedDataset.updateInstanceTargetClassValue(instancesClass1, 1);

        //run classifier after adding
        int[] testFoldLabelsAfterAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        //score dist after change dataset
/*        ScoreDistributionBasedAttributes scoreDistributionBatch = new ScoreDistributionBasedAttributes();
        TreeMap<Integer,AttributeInfo> scoreDistributionCurrentIteration = scoreDistributionBatch.getScoreDistributionBasedAttributes(
                unlabeledToMetaFeatures,labeledToMetaFeatures,
                i, evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults,
                targetClassIndexdataset.getTargetColumnIndex(),"td", properties);
        */
        //AUC after change dataset
        if (isAUC){
            AUC aucAfterAddBatch = new AUC();
            double measureAucAfterAddBatch = aucAfterAddBatch.measure
                    (testFoldLabelsAfterAdding,
                            getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions()
                                    ,1));
            int[] infoListAfter = new int[5];
            infoListAfter[0] = batch_id;
            infoListAfter[1] = expID;
            infoListAfter[2] = innerIteration;
            infoListAfter[3] = testFoldLabelsAfterAdding.length;
            infoListAfter[4] = 1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
            result.put(infoListAfter, measureAucAfterAddBatch);
//        writeToBatchScoreTbl(batch_id, expID, innerIteration, "auc_after_add_batch", measureAucAfterAddBatch, testFoldLabelsAfterAdding.length, properties);
        }
        return result;
    }


    private TreeMap<Integer,AttributeInfo> tdScoreDist(Dataset dataset
            , HashMap<Integer, List<Integer>> feature_sets, HashMap<Integer, Integer> assignedLabelsOriginalIndex
            , List<Integer> labeledTrainingSetIndices, List<Integer> unlabeledTrainingSetIndices
            , TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree, EvaluationPerIteraion unifiedDatasetEvaulationResults
            , Fold testFold, int targetClassIndex, int i, int exp_id, int batchIndex, Properties properties) throws Exception{
        TreeMap<Integer,AttributeInfo> scores = new TreeMap<>();
        ScoreDistributionBasedAttributesTdBatch scoreDist = new ScoreDistributionBasedAttributesTdBatch();

        //auc before add
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        AUC aucBeforeAddBatch = new AUC();
        int[] testFoldLabelsBeforeAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        EvaluationInfo evaluationResultsBeforeAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                new ArrayList<>(testFold.getIndices()), properties);
        double measureAucBeforeAddBatch = aucBeforeAddBatch.measure
                (testFoldLabelsBeforeAdding
                        , getSingleClassValueConfidenceScore(evaluationResultsBeforeAdding.getScoreDistributions()
                        ,1));
        AttributeInfo measureAucBeforeAddBatch_att = new AttributeInfo
                ("beforeBatchAuc", Column.columnType.Numeric, measureAucBeforeAddBatch, -1);

        //add batch to the dataset
        Dataset datasetAddedBatch = generateDatasetCopyWithBatchAdded(dataset, assignedLabelsOriginalIndex
                , labeledTrainingSetIndices,  properties);

        //labeled instances in the new dataset
        List<Integer> labeledTrainingSetIndices_cloned = new ArrayList<>(labeledTrainingSetIndices);
        labeledTrainingSetIndices_cloned.addAll(assignedLabelsOriginalIndex.keySet());

        //unlabeled instances in the new dataset
        List<Integer> unlabeledTrainingSetIndices_cloned = unlabeledTrainingSetIndices;
        unlabeledTrainingSetIndices_cloned = unlabeledTrainingSetIndices_cloned.stream().filter(line -> !labeledTrainingSetIndices_cloned.contains(line)).collect(Collectors.toList());

        //change to the labeled and unlabeled format
        Dataset labeledToMetaFeatures_td = datasetAddedBatch;
        Dataset unlabeledToMetaFeatures_td = datasetAddedBatch;
        boolean getDatasetInstancesSucc = false;
        for (int numberOfTries = 0; numberOfTries < 5 && !getDatasetInstancesSucc; numberOfTries++) {
            try{
                labeledToMetaFeatures_td = getDataSetByInstancesIndices(datasetAddedBatch,labeledTrainingSetIndices_cloned,(-1)*exp_id,batchIndex,properties);
                unlabeledToMetaFeatures_td = getDataSetByInstancesIndices(datasetAddedBatch,unlabeledTrainingSetIndices_cloned,(-1)*exp_id,batchIndex,properties);
                getDatasetInstancesSucc = true;
            }
            catch (Exception e){
                Thread.sleep(1000);
                System.out.println("failed reading file, sleep for 1 second, for try");
                getDatasetInstancesSucc = false;
            }
        }
        //eveluate results of the new dataset - 3 EvaluationInfo objects: 2 per partition + unified
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = datasetAddedBatch.replicateDatasetByColumnIndices(feature_sets.get(index));
            datasetPartitions.put(index, partition);
        }
        HashMap<Integer, EvaluationInfo> evaluationPerPartition_td = new HashMap<>();
        for (int partitionIndex : feature_sets.keySet()) {
            EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices_cloned),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices_cloned),
                    new ArrayList<>(unlabeledTrainingSetIndices), properties);
            evaluationPerPartition_td.put(partitionIndex, evaluationResults);
        }
        EvaluationInfo unifiedSetEvaluationResults_td = runClassifier(properties.getProperty("classifier"),
                dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices_cloned),
                dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices_cloned),
                new ArrayList<>(unlabeledTrainingSetIndices_cloned), properties);

        //calculate meta features
        scores = scoreDist.getScoreDistributionBasedAttributes(datasetAddedBatch
                , evaluationPerPartition_td, unifiedSetEvaluationResults_td
                , evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults
                , labeledToMetaFeatures_td, unlabeledToMetaFeatures_td
                , i, targetClassIndex, properties);

        //aud after add
        for (Map.Entry<Integer,Integer> entry : assignedLabelsOriginalIndex.entrySet()){
            int instancePos = entry.getKey();
            clonedlabeLedTrainingSetIndices.add(instancePos);
        }
        int[] testFoldLabelsAfterAdding = datasetAddedBatch.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                datasetAddedBatch.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                datasetAddedBatch.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        //AUC after change dataset
        AUC aucAfterAddBatch = new AUC();
        double measureAucAfterAddBatch = aucAfterAddBatch.measure
                (testFoldLabelsAfterAdding,
                        getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions()
                                ,1));
        AttributeInfo measureAucAfterAddBatch_att = new AttributeInfo
                ("afterBatchAuc", Column.columnType.Numeric, measureAucAfterAddBatch, -1);
        double aucDifference = measureAucAfterAddBatch - measureAucBeforeAddBatch;
        AttributeInfo aucDifference_att = new AttributeInfo
                ("BatchAucDifference", Column.columnType.Numeric, aucDifference, -1);

        //add auc before and after to the results
        scores.put(scores.size(), measureAucBeforeAddBatch_att);
        scores.put(scores.size(), measureAucAfterAddBatch_att);
        scores.put(scores.size(), aucDifference_att);
        return scores;
    }


    //all candidates for batches - for all partition and class
    private ArrayList<ArrayList<ArrayList<Integer>>> getTopCandidates(
            HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , List<Integer> unlabeledTrainingSetIndices) {
        ArrayList<ArrayList<ArrayList<Integer>>> res = new ArrayList<>();
        HashSet<Integer> uniqueInstances = new HashSet<>();
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
            for (int class_num=0; class_num<=1; class_num++){
                ArrayList<ArrayList<Integer>> result = new ArrayList<>();
                //get all candidates
                ArrayList<Integer> topInstancesCandidates = new ArrayList<>();
                TreeMap<Double, List<Integer>> topConfTree = evaluationResultsPerSetAndInteration.get(partitionIndex)
                        .getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(class_num);
                //limit to top 10% of instances
                int limitItems = (int)(evaluationResultsPerSetAndInteration.get(partitionIndex)
                        .getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(class_num).size()*0.1);
                //flatten top instances
                for(Map.Entry<Double, List<Integer>> entry : topConfTree.entrySet()) {
                    if (limitItems < 1){
                        break;
                    }
                    topInstancesCandidates.addAll(entry.getValue());
                    limitItems--;
                }
                Collections.shuffle(topInstancesCandidates);
                //select top instances - 4 per partition per class
                int countTop = 3;
                for(int top_results=0; top_results<=countTop; top_results++){
                    int relativeIndex = topInstancesCandidates.get(top_results);
                    if (!uniqueInstances.contains(relativeIndex)){
                        Integer instancePos = relativeIndex; //Integer instancePos = unlabeledTrainingSetIndices.get(relativeIndex);
                        ArrayList<Integer> relative__and_real_pos = new ArrayList<>();
                        relative__and_real_pos.add(relativeIndex);
                        relative__and_real_pos.add(instancePos);
                        result.add(relative__and_real_pos);
                        uniqueInstances.add(relativeIndex);
                    }
                    //remove duplicates
                    else{
                        countTop++;
                    }
                }
                res.add(result);
            }
        }
        return res;
    }

    // return structure:
    // [0]instancesBatchOrginalPos, [1]instancesBatchSelectedPos
    // [2]assignedLabelsOriginalIndex_0, [3]assignedLabelsOriginalIndex_1
    // [4]assignedLabelsSelectedIndex_0, [5]assignedLabelsSelectedIndex_1
    private ArrayList<ArrayList<Integer>> generateBatch(ArrayList<ArrayList<ArrayList<Integer>>> topSelectedInstancesCandidatesArr
            , HashMap<Character, int[]> pairsDict, char pair_0_0, char pair_0_1, char pair_1_0, char pair_1_1) {
        int[] pair_0_0_inx = pairsDict.get(pair_0_0);
        int[] pair_0_1_inx = pairsDict.get(pair_0_1);
        int[] pair_1_0_inx = pairsDict.get(pair_1_0);
        int[] pair_1_1_inx = pairsDict.get(pair_1_1);

        ArrayList<Integer> partition_0_class_0_ins_1 = topSelectedInstancesCandidatesArr.get(0).get(pair_0_0_inx[0]);
        ArrayList<Integer> partition_0_class_0_ins_2 = topSelectedInstancesCandidatesArr.get(0).get(pair_0_0_inx[1]);
        ArrayList<Integer> partition_0_class_1_ins_1 = topSelectedInstancesCandidatesArr.get(1).get(pair_0_1_inx[0]);
        ArrayList<Integer> partition_0_class_1_ins_2 = topSelectedInstancesCandidatesArr.get(1).get(pair_0_1_inx[1]);
        ArrayList<Integer> partition_1_class_0_ins_1 = topSelectedInstancesCandidatesArr.get(2).get(pair_1_0_inx[0]);
        ArrayList<Integer> partition_1_class_0_ins_2 = topSelectedInstancesCandidatesArr.get(2).get(pair_1_0_inx[1]);
        ArrayList<Integer> partition_1_class_1_ins_1 = topSelectedInstancesCandidatesArr.get(3).get(pair_1_1_inx[0]);
        ArrayList<Integer> partition_1_class_1_ins_2 = topSelectedInstancesCandidatesArr.get(3).get(pair_1_1_inx[1]);

        ArrayList<Integer> instancesBatchOrginalPos = new ArrayList<>();
        instancesBatchOrginalPos.add(partition_0_class_0_ins_1.get(1));
        instancesBatchOrginalPos.add(partition_0_class_0_ins_2.get(1));
        instancesBatchOrginalPos.add(partition_0_class_1_ins_1.get(1));
        instancesBatchOrginalPos.add(partition_0_class_1_ins_2.get(1));
        instancesBatchOrginalPos.add(partition_1_class_0_ins_1.get(1));
        instancesBatchOrginalPos.add(partition_1_class_0_ins_2.get(1));
        instancesBatchOrginalPos.add(partition_1_class_1_ins_1.get(1));
        instancesBatchOrginalPos.add(partition_1_class_1_ins_2.get(1));

        ArrayList<Integer> instancesBatchSelectedPos = new ArrayList<>();
        instancesBatchSelectedPos.add(partition_0_class_0_ins_1.get(0));
        instancesBatchSelectedPos.add(partition_0_class_0_ins_2.get(0));
        instancesBatchSelectedPos.add(partition_0_class_1_ins_1.get(0));
        instancesBatchSelectedPos.add(partition_0_class_1_ins_2.get(0));
        instancesBatchSelectedPos.add(partition_1_class_0_ins_1.get(0));
        instancesBatchSelectedPos.add(partition_1_class_0_ins_2.get(0));
        instancesBatchSelectedPos.add(partition_1_class_1_ins_1.get(0));
        instancesBatchSelectedPos.add(partition_1_class_1_ins_2.get(0));

        ArrayList<Integer> assignedLabelsOriginalIndex_0 = new ArrayList<>();
        assignedLabelsOriginalIndex_0.add(partition_0_class_0_ins_1.get(1));
        assignedLabelsOriginalIndex_0.add(partition_0_class_0_ins_2.get(1));
        assignedLabelsOriginalIndex_0.add(partition_1_class_0_ins_1.get(1));
        assignedLabelsOriginalIndex_0.add(partition_1_class_0_ins_2.get(1));

        ArrayList<Integer> assignedLabelsOriginalIndex_1 = new ArrayList<>();
        assignedLabelsOriginalIndex_1.add(partition_0_class_1_ins_1.get(1));
        assignedLabelsOriginalIndex_1.add(partition_0_class_1_ins_1.get(1));
        assignedLabelsOriginalIndex_1.add(partition_1_class_1_ins_1.get(1));
        assignedLabelsOriginalIndex_1.add(partition_1_class_1_ins_2.get(1));

        ArrayList<Integer> assignedLabelsSelectedIndex_0 = new ArrayList<>();
        assignedLabelsSelectedIndex_0.add(partition_0_class_0_ins_1.get(0));
        assignedLabelsSelectedIndex_0.add(partition_0_class_0_ins_2.get(0));
        assignedLabelsSelectedIndex_0.add(partition_1_class_0_ins_1.get(0));
        assignedLabelsSelectedIndex_0.add(partition_1_class_0_ins_2.get(0));

        ArrayList<Integer> assignedLabelsSelectedIndex_1 = new ArrayList<>();
        assignedLabelsSelectedIndex_1.add(partition_0_class_1_ins_1.get(0));
        assignedLabelsSelectedIndex_1.add(partition_0_class_1_ins_1.get(0));
        assignedLabelsSelectedIndex_1.add(partition_1_class_1_ins_1.get(0));
        assignedLabelsSelectedIndex_1.add(partition_1_class_1_ins_2.get(0));

        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        res.add(instancesBatchOrginalPos);
        res.add(instancesBatchSelectedPos);
        res.add(assignedLabelsOriginalIndex_0);
        res.add(assignedLabelsOriginalIndex_1);
        res.add(assignedLabelsSelectedIndex_0);
        res.add(assignedLabelsSelectedIndex_1);
        return res;
    }
}

