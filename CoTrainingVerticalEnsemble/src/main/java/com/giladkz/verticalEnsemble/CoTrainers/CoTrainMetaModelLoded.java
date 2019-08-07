package com.giladkz.verticalEnsemble.CoTrainers;
import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.MetaLearning.InstanceAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.InstancesBatchAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributes;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import weka.core.converters.ArffSaver;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class CoTrainMetaModelLoded extends CoTrainerAbstract{
    private Properties properties;
    @Override
    public void Previous_Iterations_Analysis(EvaluationPerIteraion models, Dataset training_set_data, Dataset validation_set_data, int current_iteration) {

    }

    @Override
    public String toString() {
        return "CoTrainerMetaLearning";
    }

    @Override
    public Dataset Train_Classifiers(
            HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples
            , int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration
            , String original_arff_file, int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer
            , int exp_id, String arff, int iteration, double weight_for_log, boolean use_active_learning
            , int random_seed) throws Exception {

        /*
        * This set is meta features analyzes the scores assigned to the unlabeled training set at each iteration.
        * Its possible uses include:
        * a) Providing a stopping criteria (i.e. need to go one iteration (or more) back because we're going off-course
        * b) Assisting in the selection of unlabeled samples to be added to the labeled set
        * */
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();
        InstanceAttributes instanceAttributes = new InstanceAttributes();
        InstancesBatchAttributes instancesBatchAttributes = new InstancesBatchAttributes();

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        //use sql or non for data transfer
        //String writeFile = "sql";
        String writeFile = "";
        /* We start by partitioning the dataset based on the sets of features this function receives as a parameter */
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = dataset.replicateDatasetByColumnIndices(feature_sets.get(index));
            datasetPartitions.put(index, partition);
        }

        /*
         * Randomly select the labeled instances from the training set. The remaining ones will be used as the unlabeled.
         * It is important that we use a fixed random seed for repeatability
         * */
        List<Integer> labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);

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
        //RunExperimentsOnTestSet(exp_id, iteration, -1, dataset, dataset.getTestFolds().get(0)
        //       , dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);

        ///////////////////////////////////////////////
        // And now we can begin the iterative process//
        ///////////////////////////////////////////////

        //this object saves the results for the partitioned dataset. It is of the form parition -> iteration index -> results
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();
        HashMap<Integer, ArrayList<EvaluationInfo>> evaluationResultsOnTestSet = new HashMap<>();

        //this object save the results of the runs of the unified datasets (original labeled + labeled during the co-training process).
        EvaluationPerIteraion unifiedDatasetEvaulationResults = new EvaluationPerIteraion();

        //write meta data information in groups and not one by one
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroup = new HashMap<>();
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeBatchMetaDataInGroup = new HashMap<>();
        HashMap<int[], Double> writeSampleBatchScoreInGroup = new HashMap<>();
        int writeCounterBin = 1;

        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/

            //step 1 - train the classifiers on the labeled training set and run on the unlabeled training set
            System.out.println("labaled: " + labeledTrainingSetIndices.size() + ";  unlabeled: " + unlabeledTrainingSetIndices.size() );

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

            Dataset labeledToMetaFeatures = dataset;
            Dataset unlabeledToMetaFeatures = dataset;
            int targetClassIndex = dataset.getMinorityClassIndex();
            boolean getDatasetInstancesSucc = false;
            for (int numberOfTries = 0; numberOfTries < 10 && !getDatasetInstancesSucc; numberOfTries++) {
                try{
                    labeledToMetaFeatures = getDataSetByInstancesIndices(dataset,labeledTrainingSetIndices,properties);
                    unlabeledToMetaFeatures = getDataSetByInstancesIndices(dataset,unlabeledTrainingSetIndices,properties);
                    getDatasetInstancesSucc = true;
                }catch (Exception e){
                    //Thread.sleep(1000);
                    //TimeUnit.MILLISECONDS.sleep(1500);
                    getDatasetInstancesSucc = false;
                }
            }

            TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree = new TreeMap<>(evaluationResultsPerSetAndInteration);

            //score distribution
            TreeMap<Integer,AttributeInfo> scoreDistributionCurrentIteration = new TreeMap<>();
            scoreDistributionCurrentIteration = scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults,
                    targetClassIndex, properties);


            ArrayList<ArrayList<Integer>> batchesInstancesList = new ArrayList<>();
            List<TreeMap<Integer,AttributeInfo>> instanceAttributeCurrentIterationList = new ArrayList<>();
            HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass = new HashMap<>(); //batch->instance->class
            //pick random 1000 batches of 8 instances and get meta features
            Random rnd = new Random((i + Integer.parseInt(properties.getProperty("randomSeed"))));
            System.out.println("Started generating batches");

            if (Objects.equals(properties.getProperty("batchSelection"), "smart")){
                //smart selection.
                //create: structure: relative index -> [instance_pos, label]
                ArrayList<ArrayList<ArrayList<Integer>>> topSelectedInstancesCandidatesArr = new ArrayList<>();
                topSelectedInstancesCandidatesArr = getTopCandidates(evaluationResultsPerSetAndInteration, unlabeledTrainingSetIndices);
                System.out.println("Got all top candidates");
                /*
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
                    for (int class_num=0; class_num<=1; class_num++){
                        ArrayList<ArrayList<Integer>> topSelectedInstancesCandidates = getTopCandidates(evaluationResultsPerSetAndInteration
                                , partitionIndex, unlabeledTrainingSetIndices,class_num);
                        topSelectedInstancesCandidatesArr.add(topSelectedInstancesCandidates);
                    }
                }*/
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

                                //batch meta features
                                batchesInstancesList.add(generatedBatch.get(0));
                                batchInstancePosClass.put(batchIndex, new HashMap<>(assignedLabelsOriginalIndex));
                                TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                                        unlabeledToMetaFeatures,labeledToMetaFeatures,
                                        i, evaluationResultsPerSetAndInterationTree,
                                        unifiedDatasetEvaulationResults, targetClassIndex,
                                        generatedBatch.get(1), assignedLabelsSelectedIndex, properties);

                                int[] batchInfoToWrite = new int[3];
                                batchInfoToWrite[0]=exp_id;
                                batchInfoToWrite[1]=i;
                                batchInfoToWrite[2]=batchIndex;
                                writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);

                                batchIndex++;
                            }
                        }
                    }
                }
                //end smart selection
            }
            //random selection
            else{
                for (int batchIndex = 0; batchIndex < Integer.parseInt(properties.getProperty("numOfBatchedPerIteration")); batchIndex++) {
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
                            double scoreClass0 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().get(relativeIndex)[0];
                            double scoreClass1 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().get(relativeIndex)[1];
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
                        }

                    }
                    if (class0counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))
                            && class1counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))){
                        writeInstanceMetaDataInGroup.putAll(writeInstanceMetaDataInGroupTemp);
                        batchesInstancesList.add(instancesBatchOrginalPos);
                        batchInstancePosClass.put(batchIndex, new HashMap<>(assignedLabelsOriginalIndex));
                        TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                                unlabeledToMetaFeatures,labeledToMetaFeatures,
                                i, evaluationResultsPerSetAndInterationTree,
                                unifiedDatasetEvaulationResults, targetClassIndex,
                                instancesBatchSelectedPos, assignedLabelsSelectedIndex, properties);

                        int[] batchInfoToWrite = new int[3];
                        batchInfoToWrite[0]=exp_id;
                        batchInfoToWrite[1]=i;
                        batchInfoToWrite[2]=batchIndex;
                        writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);
                    }
                    writeInstanceMetaDataInGroupTemp.clear();
                    assignedLabelsOriginalIndex.clear();
                }
            }

            System.out.println("number of batches optional to add_1: " + batchInstancePosClass.keySet().size());


            //step 2 - get the indices of the items we want to label (separately for each class)
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = new HashMap<>();
            HashMap<Integer, List<Integer>> instancesPerPartition = new HashMap<>();
            HashMap<Integer, Integer> selectedInstancesRelativeIndexes = new HashMap<>(); //index (relative) -> assigned class
            ArrayList<Integer> indicesOfAddedInstances = new ArrayList<>(); //index(original)

            //instances of selected batch by the original algorithm
            GetIndicesOfInstancesToLabelBasicRelativeIndex(dataset, instances_per_class_per_iteration, evaluationResultsPerSetAndInteration, instancesToAddPerClass, random_seed, unlabeledTrainingSetIndices, instancesPerPartition, selectedInstancesRelativeIndexes, indicesOfAddedInstances);
            for (Integer instance: selectedInstancesRelativeIndexes.keySet()){
                Integer originalInstancePos = unlabeledTrainingSetIndices.get(instance);
                Integer assignedClass = selectedInstancesRelativeIndexes.get(instance);
                TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                        unlabeledToMetaFeatures,dataset,
                        i, evaluationResultsPerSetAndInterationTree,
                        unifiedDatasetEvaulationResults, targetClassIndex,
                        originalInstancePos, assignedClass, properties);
                int[] instanceInfoToWrite = new int[5];
                instanceInfoToWrite[0]=exp_id;
                instanceInfoToWrite[1]=iteration;
                instanceInfoToWrite[2]=i;
                instanceInfoToWrite[3]=originalInstancePos;
                instanceInfoToWrite[4]= -1;
                writeInstanceMetaDataInGroup.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
            }

            TreeMap<Integer,AttributeInfo>  selectedBatchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree,
                    unifiedDatasetEvaulationResults, targetClassIndex,
                    new ArrayList<>(selectedInstancesRelativeIndexes.keySet()), selectedInstancesRelativeIndexes, properties);
            int[] batchInfoToWrite = new int[3];
            batchInfoToWrite[0]=exp_id;
            batchInfoToWrite[1]=i;
            //batchInfoToWrite[2]= -1 + iteration*(-1);
            batchInfoToWrite[2]= -1;
            writeBatchMetaDataInGroup.put(selectedBatchAttributeCurrentIterationList, batchInfoToWrite);

            System.out.println("start write to files");
            String file_instanceMetaFeatures = "";
            String file_batchMetaFeatures = "";
            String file_scoreDistFeatures = "";
            //insert all meta-learn data into SQL tables - only if writeFile=="sql"
            file_instanceMetaFeatures = insertInstanceMetaFeaturesToMetaLearnDB(writeInstanceMetaDataInGroup
                    , properties, dataset, exp_id, i, iteration, writeFile);
            file_batchMetaFeatures = insertBatchMetaFeaturesToMetaLearnDB(writeBatchMetaDataInGroup
                    , properties, dataset, exp_id, i, iteration, writeFile);
            file_scoreDistFeatures = insertScoreDistributionToMetaLearnDB(scoreDistributionCurrentIteration
                    , i, exp_id, iteration, properties, dataset, writeFile);

            ////ignore insertBatchScoreToMetaLearnDB becasue we can't use it in the model
            //insertBatchScoreToMetaLearnDB(writeSampleBatchScoreInGroup, properties);

            ///////run the python code here///////
            int selectedBatchId = -1;

            try{
                System.out.println("Start python code");
                if (Objects.equals(writeFile, "sql")) {
                    String cmd = "python /data/home/zaksg/co-train/meta-learn/meta_model_for_java.py ";
                    //String cmd = "python /Users/guyz/Documents/CoTrainingVerticalEnsemble/meta_model/venv/meta_model_for_java.py ";
                    Process pythonRun = Runtime.getRuntime().exec(
                            cmd + original_arff_file + " " + exp_id);
                    BufferedReader pythonResults = new BufferedReader(new InputStreamReader(pythonRun.getInputStream()));
                    selectedBatchId = new Integer(pythonResults.readLine()).intValue();
                }
                //csv
                else{
                    String cmd = "python /data/home/zaksg/co-train/meta-learn/meta_model_for_java_csv.py ";
                    //String cmd = "python /Users/guyz/Documents/CoTrainingVerticalEnsemble/meta_model/venv/meta_model_for_java_csv.py ";
                    Process pythonRun = Runtime.getRuntime().exec(
                            cmd + original_arff_file+" "+file_instanceMetaFeatures+" "+file_batchMetaFeatures+" "+file_scoreDistFeatures);
                    BufferedReader pythonResults = new BufferedReader(new InputStreamReader(pythonRun.getInputStream()));
                    selectedBatchId = new Integer(pythonResults.readLine()).intValue();
                }
                System.out.println("End python code");
            }catch (Exception e){
                System.out.println("Failed run python code");
                PrintWriter pw = new PrintWriter(new File("java_exception_trace.txt"));
                e.printStackTrace(pw);
                pw.close();
                break;
            }

            //if the meta-model select a different batch from the original algorithm, we need to change instancesToAddPerClass
            if (selectedBatchId > -1){
                instancesToAddPerClass.clear();
                instancesToAddPerClass = getSelectedBatchToAdd(selectedBatchId, batchInstancePosClass, evaluationResultsPerSetAndInteration, exp_id, i, iteration, properties);
            }

            //ToDo: write a function to insert DB the selected instances and their confidence/true label
            /*super.WriteInformationOnAddedItems(instancesToAddPerClass, i, exp_id
                    ,iteration,weight_for_log,instancesPerPartition, properties, dataset);*/

            //step 3 - set the class labels of the newly labeled instances to what we THINK they are
            //because the columns of the partitions are actually the columns of the original dataset,
            //there is no problem changing things only there
            for (int classIndex : instancesToAddPerClass.keySet()) {
                dataset.updateInstanceTargetClassValue(new ArrayList<>(instancesToAddPerClass.get(classIndex).keySet()), classIndex);
            }


            //step 4 - add the selected instances to the labeled training set and remove them from the unlabeled set
            //IMPORTANT: when adding the unlabeled samples, it must be with the label I ASSUME they possess.
            List<Integer> allIndeicesToAdd = new ArrayList<>();
            for (int classIndex : instancesToAddPerClass.keySet()) {
                allIndeicesToAdd.addAll(new ArrayList<Integer>(instancesToAddPerClass.get(classIndex).keySet()));
            }
            labeledTrainingSetIndices.addAll(allIndeicesToAdd);
            unlabeledTrainingSetIndices = unlabeledTrainingSetIndices.stream()
                    .filter(line -> !allIndeicesToAdd.contains(line)).collect(Collectors.toList());

            //step 5 - train the models using the current instances and apply them to the test set
            System.out.println("Meta-model results for dataset: "+original_arff_file+" scores of iteration: " + i + ": ");
            System.out.println("Selected batch id: " + selectedBatchId);


            /* instead of running this:
            * RunExperimentsOnTestSet(exp_id, iteration, i, dataset, dataset.getTestFolds().get(0)
            * , dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);
            * we can run and get ArrayList<EvaluationInfo>
            */
            evaluationResultsOnTestSet.put(i,RunExperimentsOnTestSetGetData(exp_id, iteration, i, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties));

            //truncate all tbl_meta_learn tables for the next iteration and containers
            writeInstanceMetaDataInGroup.clear();
            writeBatchMetaDataInGroup.clear();
            writeSampleBatchScoreInGroup.clear();
            batchInstancePosClass.clear();
            if (Objects.equals(writeFile, "sql")) {
                try{
                    truncateSqlTables(properties);
                }catch (Exception e){
                    System.out.println("can't truncate tables");
                    PrintWriter pw = new PrintWriter(new File("java_exception_trace.txt"));
                    e.printStackTrace(pw);
                    pw.close();
                    break;
                }
            }
            System.out.println("dataset: "+dataset.getName()+" done insert batch and run the classifier for iteration: " + i);
        }
        try{
            writeRawResultsOnTestSet(evaluationResultsOnTestSet, exp_id,num_of_iterations);
        }catch (Exception e){
            //e.printStackTrace();
            System.out.println("couldn't writeRawResultsOnTestSet");
        }
        return null;
    }

    private void writeRawResultsOnTestSet(HashMap<Integer, ArrayList<EvaluationInfo>> evaluationResultsOnTestSet
            , int exp_id, int num_of_iterations) throws Exception{

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_testset_data_per_iteration (exp_id, exp_iteration, instance, classifier, conf_class_0, conf_class_1) values (?, ?, ?, ?, ?, ?)";
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        for (int iteration = 0; iteration < num_of_iterations; iteration++) {
            ArrayList<EvaluationInfo> evals = evaluationResultsOnTestSet.get(iteration);
            for (int classifier_num = 0; classifier_num < evals.size(); classifier_num++) {
                String classifier_name;
                if (classifier_num==0){
                    classifier_name = "one classifier";
                }else if(classifier_num==1){
                    classifier_name = "partition 0 classifier";
                }else{
                    classifier_name = "partition 1 classifier";
                }
                TreeMap<Integer,double[]> scoreDist = evals.get(classifier_num).getScoreDistributions();
                for (int instance : scoreDist.keySet()) {
                    PreparedStatement preparedStmt = conn.prepareStatement(sql);
                    preparedStmt.setInt (1, exp_id);
                    preparedStmt.setInt (2, iteration);
                    preparedStmt.setInt (3, instance);
                    preparedStmt.setString(4, classifier_name);
                    preparedStmt.setDouble(5, scoreDist.get(instance)[0]);
                    preparedStmt.setDouble(6, scoreDist.get(instance)[1]);

                    preparedStmt.execute();
                    preparedStmt.close();
                }
            }
        }
        conn.close();
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
                int limitItems = 8;
                //flatten top instances
                for(Map.Entry<Double, List<Integer>> entry : topConfTree.entrySet()) {
                    if (limitItems < 1){
                        break;
                    }
                    topInstancesCandidates.addAll(entry.getValue());
                    limitItems--;
                }
                //select top instances
                int countTop = 3;
                for(int top_results=0; top_results<=countTop; top_results++){
                    int relativeIndex = topInstancesCandidates.get(top_results);
                    if (!uniqueInstances.contains(relativeIndex)){
                        Integer instancePos = unlabeledTrainingSetIndices.get(relativeIndex);
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

    //all candidates for batches - for given partition and class
    private ArrayList<ArrayList<Integer>> getTopCandidates(
            HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , int partitionIndex, List<Integer> unlabeledTrainingSetIndices
            , int class_num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        //get all candidates
        ArrayList<Integer> topInstancesCandidates = new ArrayList<>();
        TreeMap<Double, List<Integer>> topConfTree = evaluationResultsPerSetAndInteration.get(partitionIndex)
                .getLatestEvaluationInfo().getTopConfidenceInstancesPerClass(class_num);
        int limitItems = 4;
        for(Map.Entry<Double, List<Integer>> entry : topConfTree.entrySet()) {
            if (limitItems < 1){
                break;
            }
            topInstancesCandidates.addAll(entry.getValue());
            limitItems--;
        }
        for(int top_results=0; top_results<=3; top_results++){
            int relativeIndex = topInstancesCandidates.get(top_results);
            Integer instancePos = unlabeledTrainingSetIndices.get(relativeIndex);
            ArrayList<Integer> relative_pos = new ArrayList<>();
            relative_pos.add(relativeIndex);
            relative_pos.add(instancePos);
            result.add(relative_pos);
        }
        return result;
    }


    private HashMap<Integer,HashMap<Integer,Double>> getSelectedBatchToAdd(int selectedBatchId
            , HashMap<Integer, HashMap<Integer, Integer>> batchInstancePosClass
            , HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , int expID, int innerIteration, int expIteration, Properties properties) {
        //System.out.println("number of batches optional to add_2: " + batchInstancePosClass.keySet().size());
        HashMap<Integer,HashMap<Integer,Double>> selectedBatchInstanceByClass = new HashMap<>();
        HashMap<Integer,Double[]> selectedBatchInstanceClass0 = new HashMap<>();
        HashMap<Integer,Double[]> selectedBatchInstanceClass1 = new HashMap<>();
        selectedBatchInstanceByClass.put(0, new HashMap<>());
        selectedBatchInstanceByClass.put(1, new HashMap<>());

        HashMap<Integer, Integer> selectedBatchInstanceClass = batchInstancePosClass.get(selectedBatchId);
        for (Map.Entry<Integer, Integer> instance : selectedBatchInstanceClass.entrySet()){

            int assignedClass = instance.getValue();
            int instancePos = instance.getKey();
            //double confAssignedLabel_p0 = evaluationResultsPerSetAndInteration.get(0).getLatestEvaluationInfo().getScoreDistributions()[instancePos][assignedClass];
            //double confAssignedLabel_p1 = evaluationResultsPerSetAndInteration.get(1).getLatestEvaluationInfo().getScoreDistributions()[instancePos][assignedClass];
            //double confAssignedLabel = Math.max(confAssignedLabel_p0, confAssignedLabel_p1);

            if (assignedClass == 0){
                //use default confidence score, because we don't use it to insert to the labeled set
                //selectedBatchInstanceClass0.put(instancePos, new Double[]{confAssignedLabel_p0, confAssignedLabel_p1});
                selectedBatchInstanceByClass.get(0).put(instancePos, 1.0);
            }
            else{
                //selectedBatchInstanceClass1.put(instancePos, new Double[]{confAssignedLabel_p0, confAssignedLabel_p1});
                selectedBatchInstanceByClass.get(1).put(instancePos, 1.0);
            }
        }
        try{
            //writeConfSelectedTable(selectedBatchInstanceClass0, selectedBatchInstanceClass1, expID, innerIteration, expIteration, selectedBatchId, properties);
        }catch (Exception e){
            e.printStackTrace();
        }
        return selectedBatchInstanceByClass;
    }

    //ToDo: implement
    private void writeConfSelectedTable(HashMap<Integer, Double[]> selectedBatchInstanceClass0, HashMap<Integer, Double[]> selectedBatchInstanceClass1
            , int expID, int innerIteration, int expIteration, int selectedBatchId, Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Added_Samples_Confidence (exp_id, exp_iteration, inner_iteration_id, instance_pos, batch_id, label, conf_partition_0, conf_partition_1) values (?, ?, ?, ?, ?, ?, ?, ?)";
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for (Map.Entry<Integer, Double[]> entry : selectedBatchInstanceClass0.entrySet()) {
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt(1, expID);
            preparedStmt.setInt(2, expIteration);
            preparedStmt.setInt(3, innerIteration);
            preparedStmt.setInt(4, entry.getKey());
            preparedStmt.setInt(5, selectedBatchId);
            preparedStmt.setInt(6, 0);
            preparedStmt.setDouble(7, entry.getValue()[0]);
            preparedStmt.setDouble(8, entry.getValue()[1]);
            preparedStmt.execute();
            preparedStmt.close();
        }
        for (Map.Entry<Integer, Double[]> entry : selectedBatchInstanceClass1.entrySet()) {
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt(1, expID);
            preparedStmt.setInt(2, expIteration);
            preparedStmt.setInt(3, innerIteration);
            preparedStmt.setInt(4, entry.getKey());
            preparedStmt.setInt(5, selectedBatchId);
            preparedStmt.setInt(6, 1);
            preparedStmt.setDouble(7, entry.getValue()[0]);
            preparedStmt.setDouble(8, entry.getValue()[1]);
            preparedStmt.execute();
            preparedStmt.close();
        }
        conn.close();
    }


    private Dataset getDataSetByInstancesIndices (Dataset dataset, List<Integer> setIndices, Properties properties)throws Exception{

        Loader loader = new Loader();
        String tempFilePath = properties.getProperty("tempDirectory") + "temp.arff";
        Files.deleteIfExists(Paths.get(tempFilePath));
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1,-1,0,0,0,-1,true, FoldsInfo.foldType.Train);

        //generate the labeled training instances dataset
        ArffSaver s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,setIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader = new BufferedReader(new FileReader(tempFilePath));
        Dataset newDataset = loader.readArff(reader, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader.close();

        File file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted1");
        }
        return newDataset;
    }

    private HashMap<int[], Double> runClassifierOnSampledBatch(int expID, int expIteration, int innerIteration, int batch_id
            , Dataset dataset, Fold testFold, Fold trainFold, HashMap<Integer,Dataset> datasetPartitions,
                                                               HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> labeledTrainingSetIndices, Properties properties) throws Exception {

        HashMap<int[], Double> result = new HashMap<>();
        //clone the original dataset
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        //run classifier before adding - only for the first batch in the iteration
        AUC aucBeforeAddBatch = new AUC();
        int[] testFoldLabelsBeforeAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsBeforeAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        double measureAucBeforeAddBatch =
                aucBeforeAddBatch.measure(testFoldLabelsBeforeAdding
                        , getSingleClassValueConfidenceScore(evaluationResultsBeforeAdding.getScoreDistributions()
                        ,1));
        int[] infoListBefore = new int[5];
        infoListBefore[0] = batch_id;
        infoListBefore[1] = expID;
        infoListBefore[2] = innerIteration;
        infoListBefore[3] = testFoldLabelsBeforeAdding.length;
        infoListBefore[4] = -1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
        result.put(infoListBefore, measureAucBeforeAddBatch);

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
        AUC aucAfterAddBatch = new AUC();
        int[] testFoldLabelsAfterAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), new ArrayList<>(testFold.getIndices()), properties);
        double measureAucAfterAddBatch = aucAfterAddBatch.measure
                (testFoldLabelsAfterAdding, getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions()
                        ,1));
        int[] infoListAfter = new int[5];
        infoListAfter[0] = batch_id;
        infoListAfter[1] = expID;
        infoListAfter[2] = innerIteration;
        infoListAfter[3] = testFoldLabelsAfterAdding.length;
        infoListAfter[4] = 1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
        result.put(infoListAfter, measureAucAfterAddBatch);
        return result;
    }


    //insert to tbl_meta_learn_Score_Distribution_Meta_Data
    private String insertScoreDistributionToMetaLearnDB(TreeMap<Integer,AttributeInfo> scroeDistData, int expID, int expIteration,
                                                        int innerIteration, Properties properties, Dataset dataset, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);

            String sql = "insert into tbl_meta_learn_Score_Distribution_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";
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
            return "";
        }
        //csv
        else{
            String folder = "/Users/giladkatz/tmp/co_training_files/";
            String filename = "tbl_meta_learn_Score_Distribution_Meta_Data_exp_"+expIteration+"_iteration_"+innerIteration+expID+".csv";
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
            return folder+filename;
        }
    }

    //insert to tbl_meta_learn_Instances_Meta_Data
    private String insertInstanceMetaFeaturesToMetaLearnDB(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup, Properties properties, Dataset dataset, int expID, int innerIteration, int expIteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeInstanceMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_meta_learn_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";
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
                        System.out.println
                                ("failed insert instance for: (" + att_id + ", " + instanceInfo[0] + ", "
                                        + instanceInfo[1]+ ", " + instanceInfo[2] + ", " + instanceInfo[3] + ", "
                                        + instanceInfo[4] + ", " + metaFeatureName + ", " + metaFeatureValue + ")");
                    }
                }
            }
            conn.close();
            return "";
        }
        //csv
        else{
            String folder = "/Users/giladkatz/tmp/co_training_files/";
            String filename = "tbl_meta_learn_Instances_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
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

    //insert to tbl_meta_learn_Instances_Meta_Data
    private String insertInstanceMetaFeaturesToMetaLearnDB_concurrent(ConcurrentMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup, Properties properties, Dataset dataset, int expID, int innerIteration, int expIteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeInstanceMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_meta_learn_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";
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
                        System.out.println
                                ("failed insert instance for: (" + att_id + ", " + instanceInfo[0] + ", "
                                        + instanceInfo[1]+ ", " + instanceInfo[2] + ", " + instanceInfo[3] + ", "
                                        + instanceInfo[4] + ", " + metaFeatureName + ", " + metaFeatureValue + ")");
                    }
                }
            }
            conn.close();
            return "";
        }
        //csv
        else{
            String folder = "/data/home/zaksg/co-train/meta-learn/tempMetaFiles/";
            String filename = "tbl_meta_learn_Instances_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
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
    //insert to tbl_meta_learn_Batches_Meta_Data
    private String insertBatchMetaFeaturesToMetaLearnDB(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup, Properties properties, Dataset dataset, int expID, int innerIteration, int expIteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeBatchMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_meta_learn_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";
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
            return "";
        }
        //csv
        else{
            String folder = "/Users/giladkatz/tmp/co_training_files/";
            String filename = "tbl_meta_learn_Batches_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
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

    //insert to tbl_meta_learn_Batches_Meta_Data
    private String insertBatchMetaFeaturesToMetaLearnDB_concurrent(ConcurrentMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup, Properties properties, Dataset dataset, int expID, int innerIteration, int expIteration, String writeType) throws Exception{
        //sql
        if (writeType=="sql") {
            String myDriver = properties.getProperty("JDBC_DRIVER");
            String myUrl = properties.getProperty("DatabaseUrl");
            Class.forName(myDriver);
            Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

            for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeBatchMetaDataInGroup.entrySet()) {
                String sql = "insert into tbl_meta_learn_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";
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
            return "";
        }
        //csv
        else{
            String filename = "tbl_meta_learn_Batches_Meta_Data_exp_"+expID+"_iteration_"+expIteration+innerIteration+".csv";
            FileWriter fileWriter = new FileWriter(filename);
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
            return filename;
        }
    }
    //insert to tbl_meta_learn_Batchs_Score
    private void insertBatchScoreToMetaLearnDB(HashMap<int[], Double> writeSampleBatchScoreInGroup, Properties properties) throws Exception{

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        for (Map.Entry<int[], Double> outerEntry : writeSampleBatchScoreInGroup.entrySet()){
            String sql = "insert into tbl_meta_learn_Batchs_Score(att_id, batch_id, exp_id, exp_iteration, score_type, score_value, test_set_size) values (?, ?, ?, ?, ?, ?, ?)";
            Double auc = outerEntry.getValue();
            int[] info = outerEntry.getKey();
            int att_id=0;
            //insert to table
            String score_type;
            if (info[4] < 0){
                score_type = "auc_before_add_batch";
            }
            else{
                score_type = "auc_after_add_batch";
            }
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, att_id);
            preparedStmt.setInt (2, info[0]);
            preparedStmt.setInt (3, info[1]);
            preparedStmt.setInt (4, info[2]);
            preparedStmt.setString (5, score_type);
            preparedStmt.setDouble (6, auc);
            preparedStmt.setDouble (7, info[3]);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;

        }
        conn.close();
    }

    private void truncateSqlTables(Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();
        String sql_1 = "TRUNCATE tbl_meta_learn_Score_Distribution_Meta_Data;";
        stmt.executeUpdate(sql_1);
        String sql_2 = "TRUNCATE tbl_meta_learn_Instances_Meta_Data;";
        stmt.executeUpdate(sql_2);
        String sql_3 = "TRUNCATE tbl_meta_learn_Batches_Meta_Data;";
        stmt.executeUpdate(sql_3);
        conn.close();
    }



}
