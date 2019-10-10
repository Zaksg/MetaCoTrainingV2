package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.*;

import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateAverageClassificationResults;
import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateMultiplicationClassificationResults;

public abstract class CoTrainerAbstract {

    public Dataset Train_Classifiers(HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples,
                                     int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration, String original_arff_file,
                                     int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer, int exp_id, String arff,
                                     int iteration, double weight_for_log, boolean use_active_learning, int random_seed, List<Integer> labeledTrainingSet) throws Exception {
        throw new NotImplementedException();
    }

    void Previous_Iterations_Analysis(EvaluationPerIteraion models,
                                      Dataset training_set_data, Dataset validation_set_data, int current_iteration) {
        throw new NotImplementedException();
    }

    public int getClassifierID(Properties properties) throws Exception {
        switch (properties.getProperty("classifier")) {
            case "J48":
                return 1;
            default:
                throw new Exception("unidentified classifier");
        }
    }

    /**
     * Writes all types of statistics of performance on the test set to the DB
     * @param expID
     * @param expIteration
     * @param innerIteration
     * @param classificationCalculationMethod whether we use averaging, multiplication or whatever
     * @param evaluationMetric The metric whose value we record
     * @param ensembleSize
     * @param confidenceLevelValuesMap
     */
    public void writeTestSetEvaluationResults(int expID, int expIteration, int innerIteration, String classificationCalculationMethod,
                                              String evaluationMetric, int ensembleSize, HashMap<Double,Double> confidenceLevelValuesMap,
                                              Properties properties) throws Exception {

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Test_Set_Evaluation_Results (exp_id, iteration_id, inner_iteration_id, classification_calculation_method, metric_name, ensemble_size, confidence_level, value) values (?,?,?,?,?,?,?,?)";
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for (double confidenceLevel : confidenceLevelValuesMap.keySet()) {
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, expID);
            preparedStmt.setInt (2, expIteration);

            preparedStmt.setInt (3, innerIteration);
            preparedStmt.setString (4, classificationCalculationMethod);

            preparedStmt.setString (5, evaluationMetric);
            preparedStmt.setInt (6, ensembleSize);

            preparedStmt.setDouble (7, confidenceLevel);
            preparedStmt.setDouble (8, confidenceLevelValuesMap.get(confidenceLevel));
            preparedStmt.execute();
            preparedStmt.close();
        }
        conn.close();
    }

    public void WriteInformationOnAddedItems(
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass
            , int inner_iteration, int exp_id, int exp_iteration, double weight
            , HashMap<Integer, List<Integer>> instancesPerPartition, Properties properties
            , Dataset dataset) throws Exception {

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Co_Training_Added_Samples (exp_id, exp_iteration, weight, inner_iteration, classifier_id, sample_pos, presumed_class, is_correct, certainty) values (?, ?, ?, ?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for (int classIndex : instancesToAddPerClass.keySet()) {
            for (int instanceIndex: instancesToAddPerClass.get(classIndex).keySet()) {
                PreparedStatement preparedStmt = conn.prepareStatement(sql);
                preparedStmt.setInt (1, exp_id);
                preparedStmt.setInt (2, exp_iteration);
                preparedStmt.setDouble   (3, weight);
                preparedStmt.setInt(4, inner_iteration);

                int classifierIndex = -1;
                for (int classifier : instancesPerPartition.keySet()) {
                    if (instancesPerPartition.get(classifier).contains(instanceIndex)) {
                        classifierIndex = classifier;
                    }
                }
                if (classifierIndex == -1) {
                    throw new Exception("instance index not assigned to a classififer");
                }

                preparedStmt.setInt(5, classifierIndex);
                preparedStmt.setInt(6, instanceIndex);
                preparedStmt.setInt(7, classIndex);
                preparedStmt.setInt(8, (classIndex == dataset.getInstancesClassByIndex(Arrays.asList(instanceIndex))
                        .get(instanceIndex)) ? 1 : 0);
                preparedStmt.setDouble(9, instancesToAddPerClass.get(classIndex).get(instanceIndex));
                preparedStmt.execute();
                preparedStmt.close();
            }
        }
        conn.close();
    }


    /**
     * Activates Weka with the specified data and algorithm and returns the results set
     * @param classifierName
     * @param trainingSet
     * @param testSet
     * @param properties
     * @return
     * @throws Exception
     */
    public EvaluationInfo runClassifier(String classifierName, Instances trainingSet, Instances testSet,
                                        ArrayList<Integer> instanceIndices, Properties properties) throws Exception {
        try {
            Classifier classifier = getClassifier(classifierName);
            classifier.buildClassifier(trainingSet);

            //The overall classification statistics
            Evaluation evaluation;
            evaluation = new Evaluation(trainingSet);
            evaluation.evaluateModel(classifier, testSet);

            //The confidence score for each particular instance
            double[][] scoresDist = new double[testSet.size()][];
            for (int i=0; i<testSet.size(); i++) {
                Instance testInstance = testSet.get(i);
                double[] score = classifier.distributionForInstance(testInstance);
                scoresDist[i] = score;
            }

            EvaluationInfo evalInfo = new EvaluationInfo(evaluation, scoresDist, instanceIndices);

            return evalInfo;
        }
        catch (Exception ex) {
            System.out.println("problem running classifier");
        }

        return null;
    }

    /**
     * Used to obtain the requested classifier
     * @param classifier
     * @return
     * @throws Exception
     */
    private Classifier getClassifier(String classifier) throws Exception{
        switch (classifier) {
            case "J48":
                //more commonly known as C4.5
                J48 j48 = new J48();
                return j48;
            case "SVM":
                SMO svm = new SMO();
                svm.setBuildCalibrationModels(true);
                return svm;
            case "RandomForest":
                RandomForest randomForest = new RandomForest();
                return randomForest;
            case "Logistic":
                Logistic logistic= new Logistic();
                return logistic;
            case "NaiveBayes":
                NaiveBayes nb = new NaiveBayes();
                return nb;
            default:
                throw new Exception("unknown classifier");

        }
    }


    /**
     * Returns a list of training set instances which have been selected to be the labeled training set
     * @param dataset
     * @param requiredNumOfLabeledInstances
     * @param keepClassRatio should the labeled training set have the same class ratio as the dataset
     * @param randomSeed
     * @return
     */
    public List<Integer> getLabeledTrainingInstancesIndices(Dataset dataset, int requiredNumOfLabeledInstances, boolean keepClassRatio, int randomSeed) {
        List<Integer> labeledTrainingInstancesIndices = new ArrayList<>();
        Fold trainingFold = dataset.getTrainingFolds().get(0); //there should be only ONE train fold
        Random rnd = new Random(randomSeed);

        if (keepClassRatio) {
            HashMap<Integer,Double> classRatios = dataset.getClassRatios(false);
            for (int classIndex = 0; classIndex<dataset.getNumOfClasses(); classIndex++) {
                long requiredNumOfInstancesPerClass = Math.round(classRatios.get(classIndex)*requiredNumOfLabeledInstances);

                int addedInstancesCounter = 0;
                while (addedInstancesCounter < requiredNumOfInstancesPerClass) {
                    int instanceIndex = rnd.nextInt(trainingFold.getNumOfInstancesInFold());

                    if ((int)dataset.getTargetClassColumn().getColumn().getValue(instanceIndex) == classIndex) {
                        if (!labeledTrainingInstancesIndices.contains(trainingFold.getIndices().get(instanceIndex))) {
                            labeledTrainingInstancesIndices.add(trainingFold.getIndices().get(instanceIndex));
                            addedInstancesCounter++;
                        }
                    }
                }
            }
        }

        //TODO:add requirement that the labeled set will have at least XX (configurable parameter) labels of the minority class. This is needed for highly imbalanced datasets

        /*since the number of instances in the "keep ratios" scenario may not reach the required number,
          we'll use the code for the alternative scenario to randomly add additional instances */
        while (labeledTrainingInstancesIndices.size() < requiredNumOfLabeledInstances) {
            int instanceIndex = rnd.nextInt(trainingFold.getNumOfInstancesInFold());
            if (!labeledTrainingInstancesIndices.contains(trainingFold.getIndices().get(instanceIndex))) {
                labeledTrainingInstancesIndices.add(trainingFold.getIndices().get(instanceIndex));
            }
        }


        return labeledTrainingInstancesIndices;
    }

    /**
     * Returns the confidence scores for a single class (required for the AUC calcualations library we use).
     * @param confidenceScoreDistribution
     * @param index
     * @return
     */
    public double[] getSingleClassValueConfidenceScore(TreeMap<Integer,double[]> confidenceScoreDistribution, int index) {
        double[] arrayToReturn = new double[confidenceScoreDistribution.keySet().size()];
        int counter = 0;
        for (int i : confidenceScoreDistribution.keySet()) {
            arrayToReturn[counter] = confidenceScoreDistribution.get(i)[index];
            counter++;
        }
        return arrayToReturn;
    }

    public double[] getSingleClassValueConfidenceScore(TreeMap<Integer,double[]> confidenceScoreDistribution, int[] classIndices) {
        double[] arrayToReturn = new double[confidenceScoreDistribution.keySet().size()];
        int counter = 0;
        for (int i : confidenceScoreDistribution.keySet()) {
            arrayToReturn[counter] = confidenceScoreDistribution.get(i)[classIndices[i]];
            counter++;
        }
        return arrayToReturn;
    }



    /**
     * Gets the indices of the instances which the co-training algorithm would like to label for the following
     * training iteration. IMPORTANT: the indices are 0...n because of the way Weka processes the results. These
     * indices NEED TO BE CONVERTED to the "real" indices of analyzed dataset
     * @param dataset
     * @param instances_per_class_per_iteration
     * @param evaluationResultsPerSetAndInteration
     * @param instancesToAddPerClass
     */
    public void GetIndicesOfInstancesToLabelBasic(Dataset dataset, HashMap<Integer, Integer> instances_per_class_per_iteration,
                                              HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration,
                                              HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass, int randomSeed,
                                                  List<Integer> unlabeledTrainingSetIndices,
                                              HashMap<Integer, List<Integer>> instancesPerPartition) {

        List<Integer> indicesOfAddedInstances = new ArrayList<>();
        for (int classIndex=0; classIndex<dataset.getNumOfClasses(); classIndex++) {
            instancesToAddPerClass.put(classIndex, new HashMap<>());

            for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                //get all the instnaces of the current evaluation, for a given class, ordered by confidence score
                TreeMap<Double,List<Integer>> rankedItemsPerClass = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(classIndex);
                if (!instancesPerPartition.containsKey(partitionIndex)) {
                    instancesPerPartition.put(partitionIndex, new ArrayList<>());
                }

                int instancesCounter = 0;
                for (double confidenceScore : rankedItemsPerClass.keySet()) {

                    //first, if the number of instances that have this score is smaller than the number we need, just add everything
                    if (instancesCounter + rankedItemsPerClass.get(confidenceScore).size() <= instances_per_class_per_iteration.get(classIndex)) {
                        int counter = 0;
                        for (int item : rankedItemsPerClass.get(confidenceScore)) {
                            //Weka returns indices of [0-n], we need to translate it to the indices in the inlabaled training set
                            int actualIndexInUnlabaledSet = item;
                            if (!indicesOfAddedInstances.contains(actualIndexInUnlabaledSet)) {
                                instancesToAddPerClass.get(classIndex).put(actualIndexInUnlabaledSet, confidenceScore);
                                //add the instance index to the object denoting by which classifier the decision was made
                                instancesPerPartition.get(partitionIndex).add(actualIndexInUnlabaledSet);
                                indicesOfAddedInstances.add(actualIndexInUnlabaledSet);
                                counter++;
                            }
                        }
                        instancesCounter+=counter;
                    }
                    //if this is not the case, we randomly sample the number we need
                    else {
                        Random rnd = new Random(randomSeed);
                        List<Integer> testedActualIndices = new ArrayList<>();
                        while ((instancesCounter < instances_per_class_per_iteration.get(classIndex)) && (testedActualIndices.size()<rankedItemsPerClass.get(confidenceScore).size())) {
                            int pos = rnd.nextInt(rankedItemsPerClass.get(confidenceScore).size());
                            //Weka returns indices of [0-n], we need to translate it to the indices in the inlabaled training set
                            int indexToTest = rankedItemsPerClass.get(confidenceScore).get(pos);
                            //int actualIndexInUnlabaledSet = unlabeledTrainingSetIndices.get(indexToTest);
                            int actualIndexInUnlabaledSet = indexToTest;

                            if (!testedActualIndices.contains(actualIndexInUnlabaledSet)) {
                                testedActualIndices.add(actualIndexInUnlabaledSet);
                            }
                            else {
                                continue;
                            }

                            if (!indicesOfAddedInstances.contains(actualIndexInUnlabaledSet)) {
                                indicesOfAddedInstances.add(actualIndexInUnlabaledSet);

                                //we need to check that the instance has not been added by mistake as another label or already chosen
                                boolean foundMatch = false;
                                for (int classIndexIterator : instancesToAddPerClass.keySet()) {
                                    if (instancesToAddPerClass.get(classIndexIterator).containsKey(actualIndexInUnlabaledSet)) {
                                        foundMatch = true;
                                    }
                                }
                                if (!foundMatch) {
                                    instancesToAddPerClass.get(classIndex).put(actualIndexInUnlabaledSet, confidenceScore);
                                    //add the instance index to the object denoting by which classifier the decision was made
                                    instancesPerPartition.get(partitionIndex).add(actualIndexInUnlabaledSet);
                                    instancesCounter++;
                                }
                            }
                        }
                    }

                    if (instancesCounter >= instances_per_class_per_iteration.get(classIndex)) {
                        break;
                    }
                }
            }
        }
    }

    /**
     * Gets the indices of the instances which the co-training algorithm would like to label for the following
     * training iteration. IMPORTANT: the indices are 0...n because of the way Weka processes the results. These
     * indices NEED TO BE CONVERTED to the "real" indices of analyzed dataset
     * @param dataset
     * @param instances_per_class_per_iteration
     * @param evaluationResultsPerSetAndInteration
     * @param selectedInstancesRelativeIndexes
     * @param instancesToAddPerClass
     */
    public void GetIndicesOfInstancesToLabelBasicRelativeIndex(Dataset dataset, HashMap<Integer, Integer> instances_per_class_per_iteration,
                                                  HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration,
                                                  HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass, int randomSeed,
                                                  List<Integer> unlabeledTrainingSetIndices,
                                                  HashMap<Integer, List<Integer>> instancesPerPartition,
                                                  HashMap<Integer, Integer> selectedInstancesRelativeIndexes,
                                                               ArrayList<Integer> indicesOfAddedInstances) {


        for (int classIndex=0; classIndex<dataset.getNumOfClasses(); classIndex++) {
            instancesToAddPerClass.put(classIndex, new HashMap<>());

            for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                //get all the instnaces of the current evaluation, for a given class, ordered by confidence score
                TreeMap<Double,List<Integer>> rankedItemsPerClass = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(classIndex);
                if (!instancesPerPartition.containsKey(partitionIndex)) {
                    instancesPerPartition.put(partitionIndex, new ArrayList<>());
                }

                int instancesCounter = 0;
                for (double confidenceScore : rankedItemsPerClass.keySet()) {

                    //first, if the number of instances that have this score is smaller than the number we need, just add everything
                    if (instancesCounter + rankedItemsPerClass.get(confidenceScore).size() <= instances_per_class_per_iteration.get(classIndex)) {
                        int counter = 0;
                        for (int item : rankedItemsPerClass.get(confidenceScore)) {
                            //Weka returns indices of [0-n], we need to translate it to the indices in the inlabaled training set
                            //int actualIndexInUnlabaledSet = unlabeledTrainingSetIndices.get(item);
                            int actualIndexInUnlabaledSet = item;
                            if (!indicesOfAddedInstances.contains(actualIndexInUnlabaledSet)) {
                                instancesToAddPerClass.get(classIndex).put(actualIndexInUnlabaledSet, confidenceScore);
                                //add the instance index to the object denoting by which classifier the decision was made
                                instancesPerPartition.get(partitionIndex).add(actualIndexInUnlabaledSet);
                                selectedInstancesRelativeIndexes.put(item, classIndex);
                                indicesOfAddedInstances.add(actualIndexInUnlabaledSet);
                                counter++;
                            }
                        }
                        instancesCounter+=counter;
                    }
                    //if this is not the case, we randomly sample the number we need
                    else {
                        Random rnd = new Random(randomSeed);
                        List<Integer> testedActualIndices = new ArrayList<>();
                        while ((instancesCounter < instances_per_class_per_iteration.get(classIndex)) && (testedActualIndices.size()<rankedItemsPerClass.get(confidenceScore).size())) {
                            int pos = rnd.nextInt(rankedItemsPerClass.get(confidenceScore).size());
                            //Weka returns indices of [0-n], we need to translate it to the indices in the inlabaled training set
                            int indexToTest = rankedItemsPerClass.get(confidenceScore).get(pos);
                            //int actualIndexInUnlabaledSet = unlabeledTrainingSetIndices.get(indexToTest);
                            int actualIndexInUnlabaledSet = indexToTest;

                            if (!testedActualIndices.contains(actualIndexInUnlabaledSet)) {
                                testedActualIndices.add(actualIndexInUnlabaledSet);
                            }
                            else {
                                continue;
                            }

                            if (!indicesOfAddedInstances.contains(actualIndexInUnlabaledSet)) {
                                indicesOfAddedInstances.add(actualIndexInUnlabaledSet);
                                selectedInstancesRelativeIndexes.put(indexToTest, classIndex);

                                //we need to check that the instance has not been added by mistake as another label or already chosen
                                boolean foundMatch = false;
                                for (int classIndexIterator : instancesToAddPerClass.keySet()) {
                                    if (instancesToAddPerClass.get(classIndexIterator).containsKey(actualIndexInUnlabaledSet)) {
                                        foundMatch = true;
                                    }
                                }
                                if (!foundMatch) {
                                    instancesToAddPerClass.get(classIndex).put(actualIndexInUnlabaledSet, confidenceScore);
                                    //add the instance index to the object denoting by which classifier the decision was made
                                    instancesPerPartition.get(partitionIndex).add(actualIndexInUnlabaledSet);
                                    instancesCounter++;
                                }
                            }
                        }
                    }

                    if (instancesCounter >= instances_per_class_per_iteration.get(classIndex)) {
                        break;
                    }
                }
            }
        }
    }
    /**
     * This function trains models on the labeled training set and applies them to the test set. The co-training models
     * are combined using either multiplication or averaging (each result is stored in the
     * @param testFold
     * @param trainFold
     * @param datasetPartitions
     * @param labeledTrainingSetIndices
     * @throws Exception
     */
    public void RunExperimentsOnTestSet(int expID, int expIteration, int innerIteration, Dataset dataset, Fold testFold
            , Fold trainFold, HashMap<Integer,Dataset> datasetPartitions,List<Integer> labeledTrainingSetIndices
            , Properties properties) throws Exception {

        AUC auc = new AUC();
        int[] testFoldLabels = dataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsOneClassifier = runClassifier(properties.getProperty("classifier"),
                dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                dataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<Integer>(testFold.getIndices()),
                properties);

        double oneClassifierAuc = auc.measure(testFoldLabels, getSingleClassValueConfidenceScore(evaluationResultsOneClassifier.getScoreDistributions(),1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(oneClassifierAuc)){
            oneClassifierAuc = 0.0;
        }
        HashMap<Double,Double> valuesHashmapOneClass = new HashMap<>();
        valuesHashmapOneClass.put(-1.0, oneClassifierAuc);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"one_classifier","auc",-1,
                valuesHashmapOneClass,properties);


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
        HashMap<Double,Double> valuesHashmapAveraging = new HashMap<>();
        valuesHashmapAveraging.put(-1.0,averagingAUC);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"averaging","auc",-1,
                valuesHashmapAveraging,properties);

        //now we use multiplications (the same way the original co-training paper did)
        TreeMap<Integer,double[]> multiplicationClassificationResutls = calculateMultiplicationClassificationResults(evaluationResultsPerPartition,
                dataset.getNumOfClasses(), dataset.getClassRatios(false)); //we operate under the assumption that the overall ratios are known
        double multiplicationAUC = auc.measure(testFoldLabels,
                getSingleClassValueConfidenceScore(multiplicationClassificationResutls,1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(multiplicationAUC)){
            multiplicationAUC = 0.0;
        }
        HashMap<Double,Double> valuesHashmapmultiplication = new HashMap<>();
        valuesHashmapmultiplication.put(-1.0,multiplicationAUC);
        System.out.println("One classifier AUC: " + oneClassifierAuc);
        System.out.println("AVG classifier AUC: " + averagingAUC);
        System.out.println("Multiplication classifier AUC: " + multiplicationAUC);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"multiplication","auc",-1,
                valuesHashmapmultiplication,properties);

    }

    public ArrayList<EvaluationInfo> RunExperimentsOnTestSetGetData(int expID, int expIteration, int innerIteration
            , Dataset dataset, Fold testFold, Fold trainFold, HashMap<Integer,Dataset> datasetPartitions
            ,List<Integer> labeledTrainingSetIndices, Properties properties) throws Exception{

        ArrayList<EvaluationInfo> res = new ArrayList<>();
        AUC auc = new AUC();
        int[] testFoldLabels = dataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsOneClassifier = runClassifier(properties.getProperty("classifier"),
                dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                dataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                new ArrayList<Integer>(testFold.getIndices()), properties);
        res.add(evaluationResultsOneClassifier);
        double oneClassifierAuc = auc.measure(testFoldLabels, getSingleClassValueConfidenceScore(evaluationResultsOneClassifier.getScoreDistributions(),1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(oneClassifierAuc)){
            oneClassifierAuc = 0.0;
        }
        HashMap<Double,Double> valuesHashmapOneClass = new HashMap<>();
        valuesHashmapOneClass.put(-1.0, oneClassifierAuc);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"one_classifier","auc",-1,
                valuesHashmapOneClass,properties);


        //we train the models on the partitions and applying them to the test set
        HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition = new HashMap<>();
        for (int partitionIndex : datasetPartitions.keySet()) {
            EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                    datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Test,testFold.getIndices()),
                    new ArrayList<>(testFold.getIndices()), properties);
            evaluationResultsPerPartition.put(partitionIndex,evaluationResults);
            res.add(evaluationResults);
        }

        //here we use averaging to combine the classification results of the partitions
        TreeMap<Integer,double[]> averageClassificationResults = calculateAverageClassificationResults(evaluationResultsPerPartition, dataset.getNumOfClasses());
        double averagingAUC = auc.measure(testFoldLabels,
                getSingleClassValueConfidenceScore(averageClassificationResults,1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(averagingAUC)){
            averagingAUC = 0.0;
        }
        HashMap<Double,Double> valuesHashmapAveraging = new HashMap<>();
        valuesHashmapAveraging.put(-1.0,averagingAUC);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"averaging","auc",-1,
                valuesHashmapAveraging,properties);

        //now we use multiplications (the same way the original co-training paper did)
        TreeMap<Integer,double[]> multiplicationClassificationResutls = calculateMultiplicationClassificationResults(evaluationResultsPerPartition,
                dataset.getNumOfClasses(), dataset.getClassRatios(false)); //we operate under the assumption that the overall ratios are known
        double multiplicationAUC = auc.measure(testFoldLabels,
                getSingleClassValueConfidenceScore(multiplicationClassificationResutls,1));
        //when calculating the AUC for set of only one label - it returns as NaN, so this will fix it to 0
        if (Double.isNaN(multiplicationAUC)){
            multiplicationAUC = 0.0;
        }
        HashMap<Double,Double> valuesHashmapmultiplication = new HashMap<>();
        valuesHashmapmultiplication.put(-1.0,multiplicationAUC);
        System.out.println("One classifier AUC: " + oneClassifierAuc);
        System.out.println("AVG classifier AUC: " + averagingAUC);
        System.out.println("Multiplication classifier AUC: " + multiplicationAUC);
        writeTestSetEvaluationResults(expID,expIteration,innerIteration,"multiplication","auc",-1,
                valuesHashmapmultiplication,properties);
        return res;
    }
}
