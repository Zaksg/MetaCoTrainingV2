package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.Discretizers.EqualRangeBinsDiscretizer;
import com.giladkz.verticalEnsemble.FeatureSelectors.FeatureSelectorInterface;
import com.giladkz.verticalEnsemble.FeatureSelectors.RandomParitionFeatureSelector;
import com.giladkz.verticalEnsemble.MetaLearning.InstanceAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.InstancesBatchAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributesTdBatch;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import com.giladkz.verticalEnsemble.ValueFunctions.RandomValues;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;
import weka.core.Instances;
import weka.core.converters.ArffSaver;


import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.*;
import java.io.FileOutputStream;
import java.util.stream.Collectors;

import static com.giladkz.verticalEnsemble.App.InitializeFoldsInfo;
import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateAverageClassificationResults;
import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateMultiplicationClassificationResults;

public class CoTrainOneStep extends CoTrainerAbstract {
    private Properties properties;
    private Dataset _dataset;
    private HashMap<Integer,Dataset> _datasetPartitions;
    private List<Integer> _labeledTrainingSetIndices;
    private List<Integer> _unlabeledTrainingSetIndices;
    private HashMap<Integer, List<Integer>> _featureSets;
    private HashMap<Integer, HashMap<Integer, Integer>> _batchIdToInstancesMap; //batch->instance->class
    //ToDo: add object for score distribution on iterations back
    private HashMap<Integer, EvaluationPerIteraion> _evaluationResultsPerSetAndInteration;
    private EvaluationPerIteraion _unifiedDatasetEvaulationResults;
    private int _initial_labeled_set_size;


    public void getDatasetObjFromFile(String arffName, String filePrefix, Integer expId, Properties properties) throws Exception {
        //region Initialization
        //Properties properties = new Properties();
        //InputStream input = App.class.getClassLoader().getResourceAsStream("config.properties");
        int labeled_set_option = 1; //1=20%, 0=fixed 100
//        int labeled_set_option = 0; //1=20%, 0=fixed 100
        this.properties = properties;
        DiscretizerAbstract discretizer = new EqualRangeBinsDiscretizer(Integer.parseInt(this.properties.getProperty("numOfDiscretizationBins")));
        FeatureSelectorInterface featuresSelector = new RandomParitionFeatureSelector();
        ValueFunctionInterface valueFunction = new RandomValues();
        List<Integer> sizeOfLabeledTrainingSet= Arrays.asList
                (Integer.parseInt(this.properties.getProperty("initialLabeledGroup")));
        File folder = new File(this.properties.getProperty("inputFilesDirectory"));
        FoldsInfo foldsInfo = InitializeFoldsInfo();
        Loader loader = new Loader();
        File [] listOfFiles;
        File[] listOfFilesTMP = folder.listFiles();
        List<File> listFilesBeforeShuffle = Arrays.asList(listOfFilesTMP);
        Collections.shuffle(listFilesBeforeShuffle);
        listOfFiles = (File[])listFilesBeforeShuffle.toArray();
        String[] toDoDatasets = {arffName};
        //end region Initialization
        for (File file : listOfFiles) {
            if (file.isFile() && file.getName().endsWith(".arff")
                    && Arrays.asList(toDoDatasets).contains(file.getName())) {
                //int i = 0; //random seed
                int i = expId; //random seed
                BufferedReader reader = new BufferedReader(new FileReader(file.getAbsolutePath()));
                try {
                    /*step 1: object creation*/
                    //create the dataset
                    Dataset dataset = loader.readArff
                            (reader, i, null
                                    , -1, 0.7, foldsInfo);
                    //labeled set size - 20%
                    int numOfLabeledInstances;
                    if(labeled_set_option > 0){
                        numOfLabeledInstances = (int)(dataset.getIndicesOfTrainingInstances().size()*0.2);
                    }else{
                        numOfLabeledInstances = sizeOfLabeledTrainingSet.get(0);
                    }
                    this._initial_labeled_set_size = numOfLabeledInstances;
                    System.out.println("Labeled Set Size: " + numOfLabeledInstances);
                    //create the feature selection
                    HashMap<Integer, List<Integer>> featureSets = featuresSelector.Get_Feature_Sets
                            (dataset,discretizer,valueFunction,1
                                    ,2,1000
                                    ,1,0,dataset.getName()
                                    ,false, i);
                    //create the initial labeling set
                    List<Integer> labeledTrainingSet = this.getLabeledTrainingInstancesIndices
                            (dataset,numOfLabeledInstances,true,i);
                    /*step 2: data partition + run classifiers. saving objects*/
                    datasetPartitionEval(featureSets, dataset, labeledTrainingSet, arffName, i
                            , numOfLabeledInstances, 30000, filePrefix, this.properties);

                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }

    }

    public void datasetPartitionEval(
            HashMap<Integer, List<Integer>> feature_sets
            , Dataset dataset
            , List<Integer> labeledTrainingSet
            , String original_arff_file
            , int random_seed
            , int initial_number_of_labled_samples
            , int initial_unlabeled_set_size
            , String filePrefix, Properties properties) throws Exception{

        /*step 1: split the dataset and create labeled and unlabeled sets*/
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = dataset.replicateDatasetByColumnIndices(feature_sets.get(index));

            datasetPartitions.put(index, partition);
        }

        List<Integer> labeledTrainingSetIndices;
        if (labeledTrainingSet.size() > 0 ){
            labeledTrainingSetIndices = new ArrayList<>(labeledTrainingSet);
        }else{
            labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);
        }

        /*step 2: labeled and unlabeled sets*/
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

        /*step 3: run classifiers for evaluation per partitions*/
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();
        EvaluationPerIteraion unifiedDatasetEvaulationResults = new EvaluationPerIteraion();
        for (int partitionIndex : feature_sets.keySet()) {
            EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train, labeledTrainingSetIndices),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train, unlabeledTrainingSetIndices),
                    new ArrayList<>(unlabeledTrainingSetIndices), properties);

            if (!evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
            }
            evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, 0);
        }
        EvaluationInfo unifiedSetEvaluationResults = runClassifier(properties.getProperty("classifier"),
                dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices),
                new ArrayList<>(unlabeledTrainingSetIndices), properties);
        unifiedDatasetEvaulationResults.addEvaluationInfo(unifiedSetEvaluationResults, 0);


        /*write objects*/
        writeCoTrainObjects(unlabeledTrainingSetIndices, labeledTrainingSetIndices, datasetPartitions, dataset
                , evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, feature_sets, filePrefix, properties);

    }

    /* original method signature:
    * HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples
      , int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration
      , String original_arff_file, int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer
      , int exp_id, String arff, int iteration, double weight_for_log, boolean use_active_learning
      , int random_seed, List<Integer> labeledTrainingSet
    * */
    public void runOneStep(
              String datasetObjPath
            , String datasetPartitionsObjPath
            , String labeledTrainingSetIndicesObjPath
            , String unlabeledTrainingSetIndicesObjPath
            , String evaluationResultsPerSetAndInterationObjPath
            , String unifiedDatasetEvaulationResultsObjPath
            , String featureSetObjPath
            , Integer batchIdToAdd, int iteration, int exp_id
            , String filePrefix, Properties properties
        ) throws Exception {

        /*step 1: generate current dataset settings (before adding the batch)*/
        /*Read objects*/
        String folder = properties.getProperty("modelFiles");
        FileInputStream fi_dataset = new FileInputStream(new File(folder + datasetObjPath));
        FileInputStream fi_datasetPartitions = new FileInputStream(new File(folder + datasetPartitionsObjPath));
        FileInputStream fi_labeledTrainingSetIndices = new FileInputStream(new File(folder + labeledTrainingSetIndicesObjPath));
        FileInputStream fi_unlabeledTrainingSetIndices = new FileInputStream(new File(folder + unlabeledTrainingSetIndicesObjPath));
        FileInputStream fi_featureSet = new FileInputStream(new File(folder + featureSetObjPath));
        FileInputStream fi_evaluationResultsPerSetAndInteration = new FileInputStream(new File(folder + evaluationResultsPerSetAndInterationObjPath));
        FileInputStream fi_unifiedDatasetEvaulationResults = new FileInputStream(new File(folder + unifiedDatasetEvaulationResultsObjPath));
        this.properties = properties;

        ObjectInputStream oi_dataset = new ObjectInputStream(fi_dataset);
        ObjectInputStream oi_datasetPartitions = new ObjectInputStream(fi_datasetPartitions);
        ObjectInputStream oi_labeledTrainingSetIndices = new ObjectInputStream(fi_labeledTrainingSetIndices);
        ObjectInputStream oi_unlabeledTrainingSetIndices = new ObjectInputStream(fi_unlabeledTrainingSetIndices);
        ObjectInputStream oi_featureSet = new ObjectInputStream(fi_featureSet);
        ObjectInputStream oi_evaluationResultsPerSetAndInteration = new ObjectInputStream(fi_evaluationResultsPerSetAndInteration);
        ObjectInputStream oi_unifiedDatasetEvaulationResults = new ObjectInputStream(fi_unifiedDatasetEvaulationResults);

        set_dataset((Dataset) oi_dataset.readObject());
        set_datasetPartitions((HashMap<Integer,Dataset>) oi_datasetPartitions.readObject());
        set_labeledTrainingSetIndices((List<Integer>) oi_labeledTrainingSetIndices.readObject());
        set_unlabeledTrainingSetIndices((List<Integer>) oi_unlabeledTrainingSetIndices.readObject());
        set_featureSets((HashMap<Integer, List<Integer>>) oi_featureSet.readObject());
        set_evaluationResultsPerSetAndInteration(
                (HashMap<Integer, EvaluationPerIteraion>) oi_evaluationResultsPerSetAndInteration.readObject());
        set_unifiedDatasetEvaulationResults((EvaluationPerIteraion) oi_unifiedDatasetEvaulationResults.readObject());

        /*step 2: add the batch
        * Note: for the first run - the instances list to add is empty
        * */
        ArrayList<Integer> instancesToAdd;
        if (batchIdToAdd<-1){
            //empty state for the first time
            System.out.println("Starting first iteration for exp "+ exp_id);
            instancesToAdd = new ArrayList<>();
        }else{
            System.out.println("Starting iteration: "+ iteration + " for exp "+ exp_id);
            FileInputStream fi_batchIds = new FileInputStream(new File(folder + filePrefix + "batchIds.txt"));
            ObjectInputStream oi_batchIds = new ObjectInputStream(fi_batchIds);
            set_batchIdToInstancesMap((HashMap<Integer, HashMap<Integer, Integer>>)oi_batchIds.readObject());
            instancesToAdd = new ArrayList<>(this._batchIdToInstancesMap.get(batchIdToAdd).keySet());
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = getSelectedBatchToAdd(batchIdToAdd, this._batchIdToInstancesMap);
            for (int classIndex : instancesToAddPerClass.keySet()) {
                this._dataset.updateInstanceTargetClassValue(new ArrayList<>(instancesToAddPerClass.get(classIndex).keySet()), classIndex);
            }
            writeAccuracySelectedBatch(instancesToAddPerClass, this._dataset, properties, filePrefix, exp_id, iteration, batchIdToAdd);

        }
        List<Integer> newLabeledSet = get_labeledTrainingSetIndices();
        newLabeledSet.addAll(instancesToAdd);
        set_labeledTrainingSetIndices(newLabeledSet);
        set_unlabeledTrainingSetIndices(get_unlabeledTrainingSetIndices().stream().filter(line -> !instancesToAdd.contains(line)).collect(Collectors.toList()));

        /*step 2.1: run classifiers for evaluation per partitions*/
        for (int partitionIndex : this._featureSets.keySet()) {
            EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                    this._datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train, this._labeledTrainingSetIndices),
                    this._datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train, this._unlabeledTrainingSetIndices),
                    new ArrayList<>(this._unlabeledTrainingSetIndices), properties);

            if (!this._evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                this._evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
            }
            this._evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, iteration);
        }
        EvaluationInfo unifiedSetEvaluationResults = runClassifier(properties.getProperty("classifier"),
                this._dataset.generateSet(FoldsInfo.foldType.Train,this._labeledTrainingSetIndices),
                this._dataset.generateSet(FoldsInfo.foldType.Train,this._unlabeledTrainingSetIndices),
                new ArrayList<>(this._unlabeledTrainingSetIndices), properties);
        this._unifiedDatasetEvaulationResults.addEvaluationInfo(unifiedSetEvaluationResults, iteration);

        /*step 2.2: run classifiers on test set*/
        RunMeasureTestSet(exp_id, iteration, this._dataset
                , this._dataset.getTestFolds().get(0), this._dataset.getTrainingFolds().get(0)
                , this._datasetPartitions, this._labeledTrainingSetIndices, filePrefix, properties);

        /*step 3: extract initial meta features for env - score distribution based */
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroup = new HashMap<>();
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeBatchMetaDataInGroup = new HashMap<>();
        int targetClassIndex = get_dataset().getMinorityClassIndex();
        boolean getDatasetInstancesSucc = false;
        Dataset labeledToMetaFeatures = this._dataset;
        Dataset unlabeledToMetaFeatures = this._dataset;
        for (int numberOfTries = 0; numberOfTries < 10 && !getDatasetInstancesSucc; numberOfTries++) {
            try{
                labeledToMetaFeatures = getDataSetByInstancesIndices(
                        this._dataset,this._labeledTrainingSetIndices,exp_id, -2, properties);
                unlabeledToMetaFeatures = getDataSetByInstancesIndices(
                        this._dataset,this._unlabeledTrainingSetIndices,exp_id, -2, properties);
                getDatasetInstancesSucc = true;
            }catch (Exception e){
                getDatasetInstancesSucc = false;
            }
        }
        TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree = new TreeMap<>
                (this._evaluationResultsPerSetAndInteration);
        TreeMap<Integer,AttributeInfo> scoreDistributionCurrentIteration = scoreDistributionBasedAttributes
                .getScoreDistributionBasedAttributes(
                unlabeledToMetaFeatures,labeledToMetaFeatures,
                iteration, evaluationResultsPerSetAndInterationTree, this._unifiedDatasetEvaulationResults,
                targetClassIndex,"reg", properties);
        //dataset meta features added in the python code

        /*step 4: generate action space == batches cadidates and their meta features (instance based and batch based) + reward (AUC diff) */
        InstanceAttributes instanceAttributes = new InstanceAttributes();
        InstancesBatchAttributes instancesBatchAttributes = new InstancesBatchAttributes();
        ArrayList<ArrayList<Integer>> batchesInstancesList = new ArrayList<>();
        List<TreeMap<Integer,AttributeInfo>> instanceAttributeCurrentIterationList = new ArrayList<>();
        HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass = new HashMap<>(); //batch->instance->class
        //pick random 1000 batches of 8 instances and get meta features
        Random rnd = new Random((iteration + Integer.parseInt(properties.getProperty("randomSeed"))));
        System.out.println("Started generating batches");

        //create: structure: relative index -> [instance_pos, label]
        ArrayList<ArrayList<ArrayList<Integer>>> topSelectedInstancesCandidatesArr = getTopCandidates(this._evaluationResultsPerSetAndInteration, this._unlabeledTrainingSetIndices);
        System.out.println("Got all top candidates");
        //generate batches
        int batchIndex = 0;
        HashMap<Character,int[]> pairsDict = new HashMap<>();
        pairsDict.put('a',new int[]{0,1});
        pairsDict.put('b',new int[]{0,2});
        pairsDict.put('c',new int[]{0,3});
        pairsDict.put('d',new int[]{1,2});
        pairsDict.put('e',new int[]{1,3});
        pairsDict.put('f',new int[]{2,3});
        // To reduce number of batches, we use "abcde" instead of "abcdef", to get 5^4=625 batches
        for(char pair_0_0 : "abcde".toCharArray()) {
            for(char pair_0_1 : "abcde".toCharArray()) {
                for(char pair_1_0 : "abcde".toCharArray()) {
                    for(char pair_1_1 : "abcde".toCharArray()) {
                        //structure:
                        // [0]instancesBatchOrginalPos, [1]instancesBatchSelectedPos
                        // [2]assignedLabelsOriginalIndex_0, [3]assignedLabelsOriginalIndex_1
                        // [4]assignedLabelsSelectedIndex_0, [5]assignedLabelsSelectedIndex_1
                        ArrayList<ArrayList<Integer>> generatedBatch = new ArrayList<>();
                        generatedBatch = generateBatch(topSelectedInstancesCandidatesArr, pairsDict
                                ,pair_0_0, pair_0_1, pair_1_0, pair_1_1);
                        HashMap<Integer, Integer> assignedLabelsOriginalIndex = new HashMap<>();
                        HashMap<Integer, Integer> assignedLabelsSelectedIndex = new HashMap<>();
                        //instance meta features
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
                                    unlabeledToMetaFeatures,this._dataset,
                                    iteration, evaluationResultsPerSetAndInterationTree,
                                    this._unifiedDatasetEvaulationResults, targetClassIndex,
                                    instancePos, assignedClass, properties);
                            instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                            int[] instanceInfoToWrite = new int[5];
                            instanceInfoToWrite[0]=exp_id;
                            instanceInfoToWrite[1]=iteration;
                            instanceInfoToWrite[2]=iteration;
                            instanceInfoToWrite[3]=instancePos;
                            instanceInfoToWrite[4]=batchIndex;
                            writeInstanceMetaDataInGroup.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
                        }

                        //batch meta features
                        batchesInstancesList.add(generatedBatch.get(0));
                        batchInstancePosClass.put(batchIndex, new HashMap<>(assignedLabelsOriginalIndex));
                        TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                                unlabeledToMetaFeatures,labeledToMetaFeatures,
                                iteration, evaluationResultsPerSetAndInterationTree,
                                this._unifiedDatasetEvaulationResults, targetClassIndex,
                                generatedBatch.get(1), assignedLabelsSelectedIndex, properties);

                        int[] batchInfoToWrite = new int[3];
                        batchInfoToWrite[0]=exp_id;
                        batchInfoToWrite[1]=iteration;
                        batchInfoToWrite[2]=batchIndex;
                        writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);
                        
                        TreeMap<Integer,AttributeInfo> tdScoreDistributionCurrentIteration = tdScoreDist(this._dataset, this._featureSets
                                , assignedLabelsOriginalIndex, this._labeledTrainingSetIndices,this._unlabeledTrainingSetIndices
                                , evaluationResultsPerSetAndInterationTree, this._unifiedDatasetEvaulationResults
                                , this._dataset.getTestFolds().get(0), targetClassIndex, iteration, exp_id, batchIndex, properties);

                        writeBatchMetaDataInGroup.put(tdScoreDistributionCurrentIteration, batchInfoToWrite);
                        batchIndex++;
                    }
                    topSelectedInstancesCandidatesArr = getTopCandidates(this._evaluationResultsPerSetAndInteration, this._unlabeledTrainingSetIndices);
                }
                topSelectedInstancesCandidatesArr = getTopCandidates(this._evaluationResultsPerSetAndInteration, this._unlabeledTrainingSetIndices);
            }
        }
        this._batchIdToInstancesMap = batchInstancePosClass;
        System.out.println("Done Meta Data, start write files");
        //end smart selection
        writeCoTrainEnv(filePrefix, batchInstancePosClass, writeInstanceMetaDataInGroup, writeBatchMetaDataInGroup
                , scoreDistributionCurrentIteration, properties, this._dataset, exp_id, iteration);


        /*step 5: write objects again*/
        writeCoTrainObjects(get_unlabeledTrainingSetIndices(), get_labeledTrainingSetIndices()
                , get_datasetPartitions(), get_dataset(), this._evaluationResultsPerSetAndInteration
                , this._unifiedDatasetEvaulationResults, this._featureSets, filePrefix, properties);

        writeArffFilesLabalAndUnlabeled(this._dataset, this._unlabeledTrainingSetIndices, this._labeledTrainingSetIndices, iteration, exp_id
                , this._initial_labeled_set_size, topBatchesToAdd, properties);

        //close connections
        fi_dataset.close();
        fi_datasetPartitions.close();
        fi_labeledTrainingSetIndices.close();
        fi_featureSet.close();
        fi_unlabeledTrainingSetIndices.close();
        fi_evaluationResultsPerSetAndInteration.close();
        fi_unifiedDatasetEvaulationResults.close();
        oi_dataset.close();
        oi_datasetPartitions.close();
        oi_labeledTrainingSetIndices.close();
        oi_featureSet.close();
        oi_unlabeledTrainingSetIndices.close();
        oi_evaluationResultsPerSetAndInteration.close();
        oi_unifiedDatasetEvaulationResults.close();

    }

    private void RunMeasureTestSet(int expID, int iteration, Dataset dataset, Fold testFold
            , Fold trainFold, HashMap<Integer, Dataset> datasetPartitions
            , List<Integer> labeledTrainingSetIndices, String filePrefix, Properties properties) throws Exception{

        AUC auc = new AUC();
        int[] testFoldLabels = dataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsOneClassifier = runClassifier(properties.getProperty("classifier"),
                dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                dataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices())
                    , new ArrayList<Integer>(testFold.getIndices()),properties);

        double oneClassifierAuc = auc.measure(testFoldLabels, getSingleClassValueConfidenceScore(evaluationResultsOneClassifier.getScoreDistributions(),1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(oneClassifierAuc)){
            oneClassifierAuc = 0.0;
        }

        //we train the models on the partitions and applying them to the test set
        HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition = new HashMap<>();
        for (int partitionIndex : datasetPartitions.keySet()) {
            EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                    new ArrayList<Integer>(testFold.getIndices()),properties);
            evaluationResultsPerPartition.put(partitionIndex,evaluationResults);
        }

        //here we use averaging to combine the classification results of the partitions
        TreeMap<Integer,double[]> averageClassificationResults = calculateAverageClassificationResults(evaluationResultsPerPartition, dataset.getNumOfClasses());
        double averagingAUC = auc.measure(testFoldLabels,
                getSingleClassValueConfidenceScore(averageClassificationResults,1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(averagingAUC)){
            averagingAUC = 0.0;
        }

        //now we use multiplications (the same way the original co-training paper did)
        TreeMap<Integer,double[]> multiplicationClassificationResutls = calculateMultiplicationClassificationResults(evaluationResultsPerPartition,
                dataset.getNumOfClasses(), dataset.getClassRatios(false));
        double multiplicationAUC = auc.measure(testFoldLabels,
                getSingleClassValueConfidenceScore(multiplicationClassificationResutls,1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(multiplicationAUC)){
            multiplicationAUC = 0.0;
        }

        System.out.println("One classifier AUC: " + oneClassifierAuc);
        System.out.println("AVG classifier AUC: " + averagingAUC);
        System.out.println("Multiplication classifier AUC: " + multiplicationAUC);

        //write results to CSV
        String folder = properties.getProperty("modelFiles") + filePrefix;
        String filename = iteration + "_AUC_measures.csv";
        FileWriter fileWriter = new FileWriter(folder+filename);
        String fileHeader = "exp_id,iteration,one_classifier_auc,avg_auc,multiplication_auc\n";
        fileWriter.append(fileHeader);
        try {
            //insert to table
            fileWriter.append(String.valueOf(expID));
            fileWriter.append(",");
            fileWriter.append(String.valueOf(iteration));
            fileWriter.append(",");
            fileWriter.append(String.valueOf(oneClassifierAuc));
            fileWriter.append(",");
            fileWriter.append(String.valueOf(averagingAUC));
            fileWriter.append(",");
            fileWriter.append(String.valueOf(multiplicationAUC));
            fileWriter.append("\n");
        } catch (Exception e) {
            e.printStackTrace();
        }

        fileWriter.flush();
        fileWriter.close();
    }

    private void writeCoTrainEnv(
            String filePrefix, HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass
            , HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup
            , HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup
            , TreeMap<Integer, AttributeInfo> scoreDistributionCurrentIteration
            , Properties properties, Dataset dataset, int exp_id, int iteration) throws Exception{

        //csv files
        String file_instanceMetaFeatures = insertInstanceMetaFeaturesToMetaLearnDB(writeInstanceMetaDataInGroup
                , filePrefix, properties, this._dataset, exp_id, iteration, "");
        String file_batchMetaFeatures = insertBatchMetaFeaturesToMetaLearnDB(writeBatchMetaDataInGroup
                , filePrefix, properties, this._dataset, exp_id, iteration, "");
        String file_scoreDistFeatures = insertScoreDistributionToMetaLearnDB(scoreDistributionCurrentIteration
                , filePrefix, properties, this._dataset, exp_id, iteration, "");
        //object file
        String file_batchCandidates = insertBatchCandidatesDB(batchInstancePosClass
                , filePrefix, iteration, exp_id, properties, this._dataset);
        System.out.println(file_instanceMetaFeatures+" "+file_batchMetaFeatures+" "+file_scoreDistFeatures+" "+file_batchCandidates);
    }

    private void writeCoTrainObjects(
            List<Integer> unlabeledTrainingSetIndices
            , List<Integer> labeledTrainingSetIndices
            , HashMap<Integer, Dataset> datasetPartitions
            , Dataset dataset
            , HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , EvaluationPerIteraion unifiedDatasetEvaulationResults
            , HashMap<Integer, List<Integer>> featureSet
            , String filePrefix, Properties properties)
            throws IOException {

        String folder = properties.getProperty("modelFiles") + filePrefix;
        set_dataset(dataset);
        set_datasetPartitions(datasetPartitions);
        set_labeledTrainingSetIndices(labeledTrainingSetIndices);
        set_unlabeledTrainingSetIndices(unlabeledTrainingSetIndices);
        set_evaluationResultsPerSetAndInteration(evaluationResultsPerSetAndInteration);
        set_unifiedDatasetEvaulationResults(unifiedDatasetEvaulationResults);
        set_featureSets(featureSet);

        FileOutputStream f_unlabeledSet = new FileOutputStream(new File(folder + "unlabeledSet.txt"), false);
        FileOutputStream f_labeledSet = new FileOutputStream(new File(folder + "labeledSet.txt"), false);
        FileOutputStream f_datasetPartitions = new FileOutputStream(new File(folder + "datasetPartitions.txt"), false);
        FileOutputStream f_dataset = new FileOutputStream(new File(folder + "dataset.txt"), false);
        FileOutputStream f_featureSet = new FileOutputStream(new File(folder + "featureSet.txt"), false);
        FileOutputStream f_evaluationResultsPerSetAndInteration = new FileOutputStream
                (new File(folder + "evaluationResultsPerSetAndInteration.txt"), false);
        FileOutputStream f_unifiedDatasetEvaulationResults = new FileOutputStream
                (new File(folder + "unifiedDatasetEvaulationResults.txt"), false);

        ObjectOutputStream o_unlabeledSet = new ObjectOutputStream(f_unlabeledSet);
        ObjectOutputStream o_labeledSet = new ObjectOutputStream(f_labeledSet);
        ObjectOutputStream o_datasetPartitions = new ObjectOutputStream(f_datasetPartitions);
        ObjectOutputStream o_dataset = new ObjectOutputStream(f_dataset);
        ObjectOutputStream o_featureSet = new ObjectOutputStream(f_featureSet);
        ObjectOutputStream o_evaluationResultsPerSetAndInteration = new ObjectOutputStream(f_evaluationResultsPerSetAndInteration);
        ObjectOutputStream o_unifiedDatasetEvaulationResults = new ObjectOutputStream(f_unifiedDatasetEvaulationResults);

        o_unlabeledSet.writeObject(unlabeledTrainingSetIndices);
        o_labeledSet.writeObject(labeledTrainingSetIndices);
        o_datasetPartitions.writeObject(datasetPartitions);
        o_dataset.writeObject(dataset);
        o_featureSet.writeObject(featureSet);
        o_evaluationResultsPerSetAndInteration.writeObject(evaluationResultsPerSetAndInteration);
        o_unifiedDatasetEvaulationResults.writeObject(unifiedDatasetEvaulationResults);

        f_unlabeledSet.close();
        f_labeledSet.close();
        f_datasetPartitions.close();
        f_dataset.close();
        f_featureSet.close();
        f_evaluationResultsPerSetAndInteration.close();
        f_unifiedDatasetEvaulationResults.close();
        o_unlabeledSet.close();
        o_labeledSet.close();
        o_datasetPartitions.close();
        o_dataset.close();
        o_featureSet.close();
        o_evaluationResultsPerSetAndInteration.close();
        o_unifiedDatasetEvaulationResults.close();
    }

    private Dataset getDataSetByInstancesIndices(Dataset dataset, List<Integer> setIndices
            , int exp_id, int batchIndex, Properties properties)throws Exception{

        Date expDate = new Date();
        Loader loader = new Loader();
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1
                ,-1,0,0,0,-1
                ,true, FoldsInfo.foldType.Train);
        Instances indicedInstances = dataset.generateSet(FoldsInfo.foldType.Train, setIndices);
        Dataset newDataset = loader.readArff(indicedInstances, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo
                , dataset, FoldsInfo.foldType.Train, setIndices);

        return newDataset;
    }

    private TreeMap<Integer,AttributeInfo> tdScoreDist_old(Dataset dataset
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
        //scores.put(scores.size(), measureAucBeforeAddBatch_att);
        //scores.put(scores.size(), measureAucAfterAddBatch_att);
        scores.put(scores.size(), aucDifference_att);
        return scores;
    }

    private TreeMap<Integer,AttributeInfo> tdScoreDist(Dataset dataset
            , HashMap<Integer, List<Integer>> feature_sets, HashMap<Integer, Integer> assignedLabelsOriginalIndex
            , List<Integer> labeledTrainingSetIndices, List<Integer> unlabeledTrainingSetIndices
            , TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree, EvaluationPerIteraion unifiedDatasetEvaulationResults
            , Fold testFold, int targetClassIndex, int i, int exp_id, int batchIndex, Properties properties) throws Exception{
        TreeMap<Integer,AttributeInfo> scores = new TreeMap<>();
        ScoreDistributionBasedAttributesTdBatch scoreDist = new ScoreDistributionBasedAttributesTdBatch();

        //Dataset datasetAddedBatch = generateDatasetCopyWithBatchAdded(dataset, assignedLabelsOriginalIndex, labeledTrainingSetIndices,  properties);
        Dataset datasetAddedBatch = getDataSetByInstancesIndices(dataset, dataset.getIndices(), exp_id, batchIndex, properties);
        datasetAddedBatch = generateDatasetCopyWithBatchAdded(datasetAddedBatch, assignedLabelsOriginalIndex, labeledTrainingSetIndices,  properties);

        //labeled instances in the new dataset
        List<Integer> labeledTrainingSetIndices_cloned = new ArrayList<>(labeledTrainingSetIndices);
        labeledTrainingSetIndices_cloned.addAll(assignedLabelsOriginalIndex.keySet());

        //unlabeled instances in the new dataset
        //List<Integer> unlabeledTrainingSetIndices_cloned = unlabeledTrainingSetIndices;
        List<Integer> unlabeledTrainingSetIndices_cloned = new ArrayList<>(unlabeledTrainingSetIndices);
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
                /*dataset*/datasetAddedBatch.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices_cloned),
                /*dataset*/datasetAddedBatch.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices_cloned),
                new ArrayList<>(unlabeledTrainingSetIndices_cloned), properties);

        //calculate meta features
        scores = scoreDist.getScoreDistributionBasedAttributes(datasetAddedBatch
                , evaluationPerPartition_td, unifiedSetEvaluationResults_td
                , evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults
                , labeledToMetaFeatures_td, unlabeledToMetaFeatures_td
                , i, targetClassIndex, properties);

        //AUC after change dataset
        AUC aucAfterAddBatch = new AUC();
        int[] testFoldLabelsAfterAdding = dataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                datasetAddedBatch.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices_cloned),
                dataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        double measureAucAfterAddBatch = aucAfterAddBatch.measure(testFoldLabelsAfterAdding,
                        getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions(),1));
        AttributeInfo measureAucAfterAddBatch_att = new AttributeInfo
                ("afterBatchAuc", Column.columnType.Numeric, measureAucAfterAddBatch, -1);
        scores.put(scores.size(), measureAucAfterAddBatch_att);

        return scores;
    }

    private Dataset generateDatasetCopyWithBatchAdded (
            Dataset clonedDataset, HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {
        //clone the original dataset
        //Dataset clonedDataset = dataset.replicateDataset();

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
        assignedLabelsOriginalIndex_1.add(partition_0_class_1_ins_2.get(1));
        assignedLabelsOriginalIndex_1.add(partition_1_class_1_ins_1.get(1));
        assignedLabelsOriginalIndex_1.add(partition_1_class_1_ins_2.get(1));

        ArrayList<Integer> assignedLabelsSelectedIndex_0 = new ArrayList<>();
        assignedLabelsSelectedIndex_0.add(partition_0_class_0_ins_1.get(0));
        assignedLabelsSelectedIndex_0.add(partition_0_class_0_ins_2.get(0));
        assignedLabelsSelectedIndex_0.add(partition_1_class_0_ins_1.get(0));
        assignedLabelsSelectedIndex_0.add(partition_1_class_0_ins_2.get(0));

        ArrayList<Integer> assignedLabelsSelectedIndex_1 = new ArrayList<>();
        assignedLabelsSelectedIndex_1.add(partition_0_class_1_ins_1.get(0));
        assignedLabelsSelectedIndex_1.add(partition_0_class_1_ins_2.get(0));
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

    private void writeAccuracySelectedBatch(HashMap<Integer, HashMap<Integer, Double>> instancesToAddPerClass
            , Dataset dataset, Properties properties, String filePrefix
            , int exp_id, int iteration, int selectedBatchId) throws Exception {
        String folder = properties.getProperty("modelFiles") + filePrefix;
        String filename = "tbl_Selected_Batch_Analysis_exp_"+exp_id+"_iteration_"+iteration+".csv";
        FileWriter fileWriter = new FileWriter(folder+filename);
        String fileHeader = "exp_id,exp_iteration,batch_id,instance_pos,true_label,predicted_label,is_correct,confidence\n";
        fileWriter.append(fileHeader);
        int countAdded = 0;
        int countCorrect = 0;

        for (Integer label: instancesToAddPerClass.keySet()) {
            for (Map.Entry<Integer, Double> instanceScore : instancesToAddPerClass.get(label).entrySet()){
                int instancePos = instanceScore.getKey();
                double conf = instanceScore.getValue();
                int trueLabel = dataset.getInstancesClassByIndex(Arrays.asList(instancePos)).get(instancePos);
                countCorrect+=(trueLabel == label ? 1 : 0);

                fileWriter.append(String.valueOf(exp_id));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(iteration));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(selectedBatchId));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(instancePos));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(trueLabel));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(label));
                fileWriter.append(",");
                fileWriter.append(trueLabel == label ? "1" : "0");
                fileWriter.append(",");
                fileWriter.append(String.valueOf(conf));
                fileWriter.append("\n");
                countAdded++;
            }
        }
        System.out.println("Total instances added: " + countAdded + " correct: " + countCorrect + " " + (double)(countCorrect)/countAdded);
        fileWriter.flush();
        fileWriter.close();
    }

    private HashMap<Integer,HashMap<Integer,Double>> getSelectedBatchToAdd(int selectedBatchId
            , HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass) {
        HashMap<Integer,HashMap<Integer,Double>> selectedBatchInstanceByClass = new HashMap<>();
        selectedBatchInstanceByClass.put(0, new HashMap<>());
        selectedBatchInstanceByClass.put(1, new HashMap<>());
        HashMap<Integer, Integer> selectedBatchInstanceClass = batchInstancePosClass.get(selectedBatchId);
        for (Map.Entry<Integer, Integer> instance : selectedBatchInstanceClass.entrySet()){

            int assignedClass = instance.getValue();
            int instancePos = instance.getKey();
            if (assignedClass == 0){
                selectedBatchInstanceByClass.get(0).put(instancePos, 1.0);
            }
            else{
                selectedBatchInstanceByClass.get(1).put(instancePos, 1.0);
            }
        }
        return selectedBatchInstanceByClass;
    }

    private void writeArffFilesLabalAndUnlabeled(Dataset dataset, List<Integer> unlabeledTrainingSetIndices, List<Integer> labeledTrainingSetIndices
            , int i, int exp, int initial_number_of_labled_samples, int topBatchesToAdd, Properties properties) throws Exception{

        if(i < 10){
            String direct = properties.getProperty("modelFiles")+dataset.getName()+"_exp_" + exp
                    + "_initial_labels_" + initial_number_of_labled_samples + "_added_batch_" + topBatchesToAdd;
            // unlabeled file
            ArffSaver s_unlabeled= new ArffSaver();
            String tempFilePath_unlabeled = direct+"_unlabeled_iteration_"+i+".arff";
            Files.deleteIfExists(Paths.get(tempFilePath_unlabeled));
            s_unlabeled.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices));
            s_unlabeled.setFile(new File(tempFilePath_unlabeled));
            s_unlabeled.writeBatch();
            // labeled file
            ArffSaver s_labeled= new ArffSaver();
            String tempFilePath_labeled = direct+"_labeled_iteration_"+i+".arff";
            Files.deleteIfExists(Paths.get(tempFilePath_labeled));
            s_labeled.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices));
            s_labeled.setFile(new File(tempFilePath_labeled));
            s_labeled.writeBatch();
            //test set - needed only once

            if(i == 0){
                ArffSaver s_testset= new ArffSaver();
                String tempFilePath_testset = direct+"_testset.arff";
                Files.deleteIfExists(Paths.get(tempFilePath_testset));
                s_testset.setInstances(dataset.generateSet(FoldsInfo.foldType.Test,dataset.getIndicesOfTestInstances()));
                s_testset.setFile(new File(tempFilePath_testset));
                s_testset.writeBatch();
            }
        }

    }


    private String insertBatchCandidatesDB(
            HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass
            , String filePrefix, int iteration, int exp_id, Properties properties, Dataset dataset) throws Exception{
        String folder = properties.getProperty("modelFiles") + filePrefix;
        String fileName = "batchIds.txt";
        FileOutputStream f_batch = new FileOutputStream(new File(folder + fileName), false);
        ObjectOutputStream o_batch = new ObjectOutputStream(f_batch);
        o_batch.writeObject(batchInstancePosClass);
        f_batch.close();
        o_batch.close();
        return folder+fileName;
    }

    //insert to tbl_meta_learn_Instances_Meta_Data
    private String insertInstanceMetaFeaturesToMetaLearnDB(
            HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup
            , String filePrefix, Properties properties, Dataset dataset
            , int expID, int iteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            return "";
        }
        //csv
        else{
            String folder = properties.getProperty("modelFiles") + filePrefix;

            //String filename = "tbl_meta_learn_Instances_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
            String filename = iteration + "_Instances_Meta_Data.csv";
            FileWriter fileWriter = new FileWriter(folder+filename);
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
            return folder+filename;
        }
    }

    private String insertBatchMetaFeaturesToMetaLearnDB(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup
            , String filePrefix, Properties properties, Dataset dataset, int expID, int iteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            return "";
        }
        //csv
        else{
            String folder = properties.getProperty("modelFiles") + filePrefix;
            //String filename = "tbl_meta_learn_Batches_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
            String filename = iteration + "_Batches_Meta_Data.csv";
            FileWriter fileWriter = new FileWriter(folder+filename);
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
            return folder+filename;
        }
    }

    private String insertScoreDistributionToMetaLearnDB(TreeMap<Integer, AttributeInfo> scroeDistData, String filePrefix
            , Properties properties, Dataset dataset, int expID, int expIteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            return "";
        }
        //csv
        else{
            String folder = properties.getProperty("modelFiles") + filePrefix;
            //String filename = "tbl_meta_learn_Score_Distribution_Meta_Data_exp_"+expIteration+"_iteration_"+innerIteration+expID+".csv";
            String filename = expIteration + "_Score_Distribution_Meta_Data.csv";
            FileWriter fileWriter = new FileWriter(folder+filename);
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
                fileWriter.append(String.valueOf(expIteration));
                fileWriter.append(",");
                fileWriter.append(metaFeatureName);
                fileWriter.append(",");
                fileWriter.append(String.valueOf(metaFeatureValue));
                fileWriter.append("\n");
                att_id++;
            }
            fileWriter.flush();
            fileWriter.close();
            return folder+filename;
        }
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
                //int limitItems_group = (int)(evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(class_num).size()*0.1);
                int limitItems = (int)(evaluationResultsPerSetAndInteration.get(partitionIndex)
                        .getLatestEvaluationInfo().getScoreDistributions().size()*0.1);
                //flatten top instances
                for(Map.Entry<Double, List<Integer>> entry : topConfTree.entrySet()) {

                    for(Integer candidate: entry.getValue()){
                        if (limitItems < 1){
                            break;
                        }
                        else{
                            topInstancesCandidates.add(candidate);
                            limitItems--;
                        }
                    }
                    //topInstancesCandidates.addAll(entry.getValue());
                    //limitItems--;
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

    public Dataset get_dataset() {
        return _dataset;
    }

    public void set_dataset(Dataset _dataset) {
        this._dataset = _dataset;
    }

    public HashMap<Integer, Dataset> get_datasetPartitions() {
        return _datasetPartitions;
    }

    public void set_datasetPartitions(HashMap<Integer, Dataset> _datasetPartitions) {
        this._datasetPartitions = _datasetPartitions;
    }

    public List<Integer> get_labeledTrainingSetIndices() {
        return _labeledTrainingSetIndices;
    }

    public void set_labeledTrainingSetIndices(List<Integer> _labeledTrainingSetIndices) {
        this._labeledTrainingSetIndices = _labeledTrainingSetIndices;
    }

    public List<Integer> get_unlabeledTrainingSetIndices() {
        return _unlabeledTrainingSetIndices;
    }

    public void set_unlabeledTrainingSetIndices(List<Integer> _unlabeledTrainingSetIndices) {
        this._unlabeledTrainingSetIndices = _unlabeledTrainingSetIndices;
    }

    public HashMap<Integer, EvaluationPerIteraion> get_evaluationResultsPerSetAndInteration() {
        return _evaluationResultsPerSetAndInteration;
    }

    public void set_evaluationResultsPerSetAndInteration(HashMap<Integer, EvaluationPerIteraion> _evaluationResultsPerSetAndInteration) {
        this._evaluationResultsPerSetAndInteration = _evaluationResultsPerSetAndInteration;
    }

    public EvaluationPerIteraion get_unifiedDatasetEvaulationResults() {
        return _unifiedDatasetEvaulationResults;
    }

    public void set_unifiedDatasetEvaulationResults(EvaluationPerIteraion _unifiedDatasetEvaulationResults) {
        this._unifiedDatasetEvaulationResults = _unifiedDatasetEvaulationResults;
    }

    public HashMap<Integer, List<Integer>> get_featureSets() {
        return _featureSets;
    }

    public void set_featureSets(HashMap<Integer, List<Integer>> _featureSets) {
        this._featureSets = _featureSets;
    }

    public void set_batchIdToInstancesMap(HashMap<Integer, HashMap<Integer, Integer>> _batchIdToInstancesMap) {
        this._batchIdToInstancesMap = _batchIdToInstancesMap;
    }

    public HashMap<Integer, HashMap<Integer, Integer>> get_batchIdToInstancesMap(){return _batchIdToInstancesMap;}
}
