package com.giladkz.verticalEnsemble;

import com.giladkz.verticalEnsemble.CoTrainers.*;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Data.EvaluationPerIteraion;
import com.giladkz.verticalEnsemble.Data.FoldsInfo;
import com.giladkz.verticalEnsemble.Data.Loader;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.Discretizers.EqualRangeBinsDiscretizer;
import com.giladkz.verticalEnsemble.FeatureSelectors.FeatureSelectorInterface;
import com.giladkz.verticalEnsemble.FeatureSelectors.RandomParitionFeatureSelector;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import com.giladkz.verticalEnsemble.ValueFunctions.RandomValues;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.pmml.Array;

import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class App 
{
    public static void main( String[] args ) throws Exception
    {
        //Configurations loading
        Properties properties = new Properties();
        InputStream input = App.class.getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        //RL initiation - for the python code - first step of the RL framework
        /*test zone*/
        /*
        String folder_2 = properties.getProperty("modelFiles");
        FileInputStream fi_evaluationResultsPerSetAndInteration = new FileInputStream(new File(folder_2+"145475_german_credit_evaluationResultsPerSetAndInteration.txt"));
        ObjectInputStream oi_evaluationResultsPerSetAndInteration = new ObjectInputStream(fi_evaluationResultsPerSetAndInteration);
        TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree = new TreeMap<>((HashMap<Integer, EvaluationPerIteraion>) oi_evaluationResultsPerSetAndInteration.readObject());
        evaluationResultsPerSetAndInterationTree.size();
        */
        /*run instruction for python: python <>.py init <dataset>.arff filePrefix*/
        if (args.length > 0 && Objects.equals(args[0], "init")){
            String datasetFile = args[1];
            String filePrefix = args[2];
            CoTrainOneStep coTrainer_oneStep_init = new CoTrainOneStep();
            coTrainer_oneStep_init.getDatasetObjFromFile(datasetFile, filePrefix, properties);
        }
        //RL iteration - for the python code - second and loop step of the RL framework
        /*run instruction for python: python <>.py iteration prefix
            batchId iteration expId*/
        else if (args.length > 0 && Objects.equals(args[0], "iteration")){
            CoTrainOneStep coTrainer_oneStep_iteration = new CoTrainOneStep();
            String filePrefix = args[1];
            String ds = filePrefix + "dataset.txt";
            String ds_pt = filePrefix + "datasetPartitions.txt";
            String labeled = filePrefix + "labeledSet.txt";
            String unlabeled = filePrefix + "unlabeledSet.txt";
            String eval = filePrefix + "evaluationResultsPerSetAndInteration.txt";
            String unifi = filePrefix + "unifiedDatasetEvaulationResults.txt";
            String featues = filePrefix + "featureSet.txt";

            Integer batchId = Integer.parseInt(args[2]);
            Integer iteration = Integer.parseInt(args[3]);
            Integer expId = Integer.parseInt(args[4]);

            coTrainer_oneStep_iteration.runOneStep(ds, ds_pt, labeled, unlabeled, eval, unifi, featues, batchId, iteration, expId, filePrefix, properties);
        }
        //Meta model / original co train framework
        else {
            Loader loader = new Loader();
            buildDBTables(properties);
            buildMetaLearnDBTables(properties);
            //region Initialization
            DiscretizerAbstract discretizer = new EqualRangeBinsDiscretizer(Integer.parseInt(properties.getProperty("numOfDiscretizationBins")));
            FeatureSelectorInterface featuresSelector = new RandomParitionFeatureSelector();
            ValueFunctionInterface valueFunction = new RandomValues();
            CoTrainerAbstract coTrainerMetaModelGeneration = new CoTrainingMetaLearning();
            CoTrainerAbstract coTrainer_original = new CoTrainerOriginal();
            CoTrainerAbstract coTrainer_meta_model = new CoTrainMetaModelLoded();
            List<Integer> sizeOfLabeledTrainingSet= Arrays.asList(Integer.parseInt(properties.getProperty("initialLabeledGroup")));
            //endregion

            File folder = new File(properties.getProperty("inputFilesDirectory"));
            FoldsInfo foldsInfo = InitializeFoldsInfo();

            File[] listOfFiles;
            File[] listOfFilesTMP = folder.listFiles();
            List<File> listFilesBeforeShuffle = Arrays.asList(listOfFilesTMP);
            Collections.shuffle(listFilesBeforeShuffle);
            listOfFiles = (File[]) listFilesBeforeShuffle.toArray();
            List<String> labeledTrainingSet = Arrays.asList("100", "108", "116", "130", "140", "160", "180", "200", "250");
//            List<String> labeledTrainingSet = Arrays.asList("100");
            List<String> addedBatches = Arrays.asList("1", "3", "5", "10");
//            List<String> addedBatches = Arrays.asList("0");


            String[] toDoDatasets = {"german_credit.arff"};
            /*String[] toDoDatasets = {
                    "german_credit.arff"
                    , "puma8NH.arff"
                    , "puma32H.arff"
                    , "contraceptive.arff"
                    , "diabetes.arff"
                    , "php0iVrYT.arff"
                    , "php7KLval.arff"
                    , "seismic-bumps.arff"
            };*/
            /*String[] toDoDatasets = {
                    "ailerons.arff"
                    , "bank-full.arff"
                    , "cardiography_new.arff"
                    , "contraceptive.arff"
                    , "cpu_act.arff"
                    , "delta_elevators.arff"
                    , "diabetes.arff"
                    , "german_credit.arff"
                    , "ionosphere.arff"
                    , "kc2.arff"
                    , "mammography.arff"
                    , "page-blocks_new.arff"
                    , "php0iVrYT.arff"
                    , "php7KLval.arff"
                    , "php8Mz7BG.arff"
                    , "php9xWOpn.arff"
                    , "php50jXam.arff"
                    , "phpelnJ6y.arff"
                    , "phpOJxGL9.arff"
                    , "puma8NH.arff"
                    , "puma32H.arff"
                    , "seismic-bumps.arff"
                    , "space_ga.arff"
                    , "spambase.arff"
                    , "wind.arff"};*/
            if (args.length > 0) {
                toDoDatasets[0] = null;
                toDoDatasets = args;
            }
            //String[] doneDatasets = {"german_credit.arff"};

            //double auc_check = checkAucClac();
            for (File file : listOfFiles) {
                if (file.isFile() && file.getName().endsWith(".arff") /*&& !Arrays.asList(doneDatasets).contains(file.getName())*/
                        && Arrays.asList(toDoDatasets).contains(file.getName())) {

                    for (String labelSetSize: labeledTrainingSet/*int numOfLabeledInstances : sizeOfLabeledTrainingSet*/) {
                        for (String topBatches : addedBatches){
                            int numOfLabeledInstances = Integer.parseInt(labelSetSize);
                            int topBatchesSelection = Integer.parseInt(topBatches);
                            //properties.setProperty("initialLabeledGroup", labelSetSize);
                            //properties.setProperty("topBatchSelection", topBatches);
                            int numOfRuns = Integer.parseInt(properties.getProperty("numOfrandomSeeds"));
                            ArrayList<Runnable> tasks = new ArrayList<>();
                            ExecutorService executorService = Executors.newFixedThreadPool(numOfRuns);
                            ArrayList<ArrayList<Integer>> exp_ids = getExpIds(numOfRuns, file.getName(), coTrainer_original.toString(), featuresSelector.toString(), valueFunction.toString(), discretizer.toString(), Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), numOfLabeledInstances, properties);
                            //runDatasetSeed(0, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties);

                            for (int task_i = 0; task_i < numOfRuns; task_i++) {
                                final int taks_index = task_i;
                                Runnable task_temp = () -> {
                                    try {
                                        runDatasetSeed(taks_index, file, numOfLabeledInstances, coTrainer_original
                                                , coTrainer_meta_model, coTrainerMetaModelGeneration
                                                , discretizer, featuresSelector, valueFunction, loader, foldsInfo
                                                , properties, exp_ids.get(taks_index).get(0), exp_ids.get(taks_index).get(1), topBatchesSelection);
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                    }
                                };
                                tasks.add(task_temp);
                            }
                            for (int task_i = 0; task_i < numOfRuns; task_i++) {
                                executorService.submit(tasks.get(task_i));
                            }
                            executorService.shutdownNow();
                        }
                    }
                }
            }
        }

    }

    private static ArrayList<ArrayList<Integer>> getExpIds(int runsPerDS, String arff_name, String co_trainer, String feature_selector,
                                                           String value_function, String discretizer, int num_of_training_iterations
            , int labeled_training_set_size, Properties properties) throws Exception{
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        for (int i = 0; i < runsPerDS; i++) {
            ArrayList<Integer> temp = new ArrayList<>();
            int expID_original = getNewExperimentID(arff_name,co_trainer,feature_selector,value_function, discretizer, num_of_training_iterations,labeled_training_set_size, properties);
            int expID_meta_model = setNextExperimentID(expID_original, arff_name,co_trainer,feature_selector,value_function, discretizer, num_of_training_iterations,labeled_training_set_size, properties);
            temp.add(expID_original);
            temp.add(expID_meta_model);
            res.add(temp);
        }
        return res;
    }

    private static void runDatasetSeed(int i, File file, int numOfLabeledInstances, CoTrainerAbstract coTrainer_original, CoTrainerAbstract coTrainer_meta_model, CoTrainerAbstract coTrainerMetaModelGeneration, DiscretizerAbstract discretizer, FeatureSelectorInterface featuresSelector, ValueFunctionInterface valueFunction, Loader loader, FoldsInfo foldsInfo, Properties properties, Integer expID_original, Integer expID_meta_model, int topBatchesSelection) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(file.getAbsolutePath()));
        try{
            Dataset dataset = loader.readArff(reader, i, null, -1, 0.7, foldsInfo);
            Dataset dataset_meta_model = dataset.replicateDataset();
            //let the co-training begin
            //a) generate the feature sets
            HashMap<Integer, List<Integer>> featureSets = featuresSelector.Get_Feature_Sets(dataset,discretizer,valueFunction,1,2,1000,1,0,dataset.getName(),false, i);
            //writeFeatureSelectionToFile(featureSets, dataset, i, properties);

            //a) generate the labeled sets
            List<Integer> labeledTrainingSet = coTrainer_original.getLabeledTrainingInstancesIndices(dataset,numOfLabeledInstances,true,i);

            //meta model generation
            /*Dataset finalDataset_otiginal = coTrainerMetaModelGeneration.Train_Classifiers(featureSets,dataset,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), getNumberOfNewInstancesPerClassPerTrainingIteration(dataset.getNumOfClasses(), properties),file.getAbsolutePath(),30000, 1, discretizer, expID_original,"test",0,0,false, i, labeledTrainingSet);*/
            //original
            Dataset finalDataset_otiginal = coTrainer_original.Train_Classifiers(featureSets,dataset,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), getNumberOfNewInstancesPerClassPerTrainingIteration(dataset.getNumOfClasses(), properties),file.getAbsolutePath(),12000/*30000*/, 1, discretizer, expID_original,"test",0,0,false, i, labeledTrainingSet, 0);
            if(coTrainer_original.getStop_exp()){
                throw new Exception("High baseline");
            }


            System.out.println("Original model done with exp id: "+expID_original+". Start meta model with exp id: "+ expID_meta_model);
            //meta model selection
            //coTrainer_meta_model.setTopBatchesToAdd(topBatchesSelection);
            Dataset finalDataset_meta_model = coTrainer_meta_model.Train_Classifiers(featureSets,dataset_meta_model,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), getNumberOfNewInstancesPerClassPerTrainingIteration(dataset_meta_model.getNumOfClasses(), properties),file.getAbsolutePath(),12000/*30000*/, 1, discretizer, expID_meta_model,"test",0,0,false, i, labeledTrainingSet, topBatchesSelection);
            Date experimentEndDate = new Date();
            System.out.println(experimentEndDate.toString() + " Experiment ended");
        }catch (Exception e){
            e.printStackTrace();
            PrintWriter pw = null;
            try {
                pw = new PrintWriter(new File("java_exception_trace.txt"));
            } catch (FileNotFoundException e1) {
                e1.printStackTrace();
            }
            e.printStackTrace(pw);
            pw.close();
            System.out.println("Failed running dataset: " + file.getName());
        }
    }

    private static void writeFeatureSelectionToFile(HashMap<Integer, List<Integer>> featureSets, Dataset dataset, int i, Properties properties) throws Exception {
        String filename = properties.getProperty("tempDirectory")+dataset.getName()+"_seed_"+i+"_feature_partition.txt";
        FileWriter fileWriter = new FileWriter(filename);
        String fileHeader = "partition,featureNum\n";
        fileWriter.append(fileHeader);

        for (Integer partition: featureSets.keySet()) {
            for (Integer featureNum: featureSets.get(partition)){
                fileWriter.append(String.valueOf(partition));
                fileWriter.append(",");
                fileWriter.append(String.valueOf(featureNum));
                fileWriter.append("\n");
            }
        }
        fileWriter.flush();
        fileWriter.close();
    }

    private static double checkAucClac() throws FileNotFoundException {
        AUC auc = new AUC();
        ArffLoader loader1 = new ArffLoader();
        ArffLoader loader2 = new ArffLoader();
        try{
            loader1.setFile(new File("/Users/guyz/Documents/CoTrainingVerticalEnsemble/weka testing/german_credit_100.arff"));
            loader2.setFile(new File("/Users/guyz/Documents/CoTrainingVerticalEnsemble/weka testing/german_credit_rest.arff"));
            Instances train = loader1.getDataSet();
            Instances test = loader2.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            int[] truth = new int[test.size()];
            for (int i = 0; i < test.size(); i++) {
                truth[i] = (int)test.get(i).classValue();
            }
            
            
            RandomForest randomForest = new RandomForest();
            randomForest.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            double[] ab = new double[test.size()];
            for (int i=0; i<test.size(); i++) {
                Instance testInstance = test.get(i);
                double[] score = randomForest.distributionForInstance(testInstance);
                ab[i] = score[1];
            }
            double[] a = eval.evaluateModel(randomForest, test);
            ThresholdCurve tc = new ThresholdCurve();
            int classIndex = 1;
            Instances result = tc.getCurve(eval.predictions(), classIndex);
            double auc_code = auc.measure(truth, ab);
            double auc_weka = tc.getROCArea(result);
            return auc_code - auc_weka;
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return 0.0;
    }

    private static void buildDBTables(Properties properties)  throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();
        String sqlTbl1 = "CREATE TABLE if not exists tbl_Batches_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl1);
        String sqlTbl2 = "CREATE TABLE if not exists tbl_Batchs_Score (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  score_type varchar(500) NOT NULL," +
                "  score_value double DEFAULT NULL," +
                "  test_set_size double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,score_type)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl2);

        String sqlTbl3 = "CREATE TABLE if not exists tbl_Co_Training_Added_Samples (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  weight float NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  classifier_id int(11) NOT NULL," +
                "  sample_pos int(11) NOT NULL," +
                "  presumed_class int(11) DEFAULT NULL," +
                "  is_correct tinyint(4) DEFAULT NULL," +
                "  certainty float DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,weight,exp_iteration,sample_pos,classifier_id,inner_iteration)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl3);

        String sqlTbl4 = "CREATE TABLE if not exists tbl_Dataset (" +
                "  dataset_id int(11) NOT NULL," +
                "  dataset_name varchar(500) DEFAULT NULL," +
                "  arff_name varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (dataset_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl4);

        String sqlTbl5 = "CREATE TABLE if not exists tbl_Dataset_Meta_Data (" +
                "  dataset_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) DEFAULT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (dataset_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl5);

        String sqlTbl6 = "CREATE TABLE if not exists tbl_Experiments (" +
                "  exp_id int(11) NOT NULL," +
                "  arff_name varchar(500) DEFAULT NULL," +
                "  start_date datetime DEFAULT NULL," +
                "  co_trainer varchar(500) DEFAULT NULL," +
                "  feature_selector varchar(500) DEFAULT NULL," +
                "  value_function varchar(500) DEFAULT NULL," +
                "  discretizer varchar(500) DEFAULT NULL," +
                "  num_of_training_iterations int(11) DEFAULT NULL," +
                "  classifier varchar(500) DEFAULT NULL," +
                "  labeled_training_set_size int(11) DEFAULT NULL," +
                "  item_insertion_policy varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (exp_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl6);

        String sqlTbl7 = "CREATE TABLE if not exists tbl_Instance_In_Batch (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration_id,batch_id,instance_pos)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl7);

        String sqlTbl8 = "CREATE TABLE if not exists tbl_Instances_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,instance_pos,meta_feature_name,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl8);

        String sqlTbl9 = "CREATE TABLE if not exists tbl_Meta_Data_Added_Batches (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  total_batch_size int(11) DEFAULT NULL," +
                "  num_of_class_1_assigned_labels int(11) DEFAULT NULL," +
                "  num_of_class_2_assigned_labels int(11) DEFAULT NULL," +
                "  averaging_score_absolue double DEFAULT NULL," +
                "  multiplication_score_absolute double DEFAULT NULL," +
                "  unified_score_absolute double DEFAULT NULL," +
                "  averaging_score_delta double DEFAULT NULL," +
                "  multiplication_score_delta double DEFAULT NULL," +
                "  unified_score_delta double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl9);

        String sqlTbl10 = "CREATE TABLE if not exists tbl_Meta_Data_Batch_Info (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  assigned_class int(11) DEFAULT NULL," +
                "  true_class int(11) DEFAULT NULL," +
                "  classifier_1_score double DEFAULT NULL," +
                "  classifier_2_score double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration,batch_id,instance_pos)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl10);

        String sqlTbl11 = "CREATE TABLE if not exists tbl_Score_Distribution_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl11);

        String sqlTbl12 = "CREATE TABLE if not exists tbl_Test_Set_Evaluation_Results (" +
                "  exp_id int(11) NOT NULL," +
                "  iteration_id int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  classification_calculation_method varchar(500) NOT NULL," +
                "  metric_name varchar(500) NOT NULL," +
                "  ensemble_size int(11) NOT NULL," +
                "  confidence_level double NOT NULL," +
                "  value double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,iteration_id,inner_iteration_id,classification_calculation_method,metric_name,ensemble_size,confidence_level)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl12);
        conn.close();

        System.out.println("databases created");
    }

    private static void buildMetaLearnDBTables(Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();

        String sqlTbl_1 = "CREATE TABLE if not exists tbl_meta_learn_Batches_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_1);

        String sqlTbl_2 = "CREATE TABLE if not exists tbl_meta_learn_Batchs_Score (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  score_type varchar(500) NOT NULL," +
                "  score_value double DEFAULT NULL," +
                "  test_set_size double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,score_type)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_2);

        String sqlTbl_3 = "CREATE TABLE if not exists tbl_meta_learn_Instances_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,instance_pos,meta_feature_name,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_3);

        String sqlTbl_4 = "CREATE TABLE if not exists tbl_meta_learn_Score_Distribution_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_4);

        conn.close();

        System.out.println("meta learn databases created");
    }

    /**
     * Initializes the object containing the information regarding how the original dataset need to be partitioned into train, validaion
     * and test sets. Also determines the number of folds for each type.
     * @return
     * @throws Exception
     */
    public static FoldsInfo InitializeFoldsInfo() throws Exception {
        FoldsInfo fi = new FoldsInfo(1,0,1,0.7,-1,0,0,0.3,-1,true, FoldsInfo.foldType.Test);
        return fi;
    }

    private static HashMap<Integer,Integer> getNumberOfNewInstancesPerClassPerTrainingIteration(int numOfClasses, Properties properties) {
        HashMap<Integer,Integer> mapToReturn = new HashMap<>();
        for (int i=0; i<numOfClasses; i++) {
            mapToReturn.put(i, Integer.parseInt(properties.getProperty("numOfInstancesToAddPerIterationPerClass")));
        }
        return mapToReturn;
    }

    private static int getNewExperimentID(String arff_name, String co_trainer, String feature_selector,
                                      String value_function, String discretizer, int num_of_training_iterations, int labeled_training_set_size, Properties properties) throws Exception {
        int exp_id;
        //tbl_Clustering_Runs
        String sql = "select IFNULL ( max(exp_id), 0 ) as idx from tbl_Experiments";

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery(sql);
        if (rs.next()){
            exp_id = rs.getInt("idx");
        }
        else {
            throw new Exception("no run id created");
        }
        rs.close();
        stmt.close();

        Date date = new Date();
        sql = "insert into tbl_Experiments (exp_id, arff_name, start_date, co_trainer, feature_selector, value_function, discretizer, num_of_training_iterations, classifier, labeled_training_set_size,item_insertion_policy) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        PreparedStatement preparedStmt = conn.prepareStatement(sql);
        preparedStmt.setInt (1, exp_id+1);
        preparedStmt.setString (2, arff_name);
        preparedStmt.setTimestamp   (3, new java.sql.Timestamp(new java.util.Date().getTime()));
        preparedStmt.setString(4, co_trainer);
        preparedStmt.setString(5, feature_selector);
        preparedStmt.setString(6, value_function);
        preparedStmt.setString(7, discretizer);
        preparedStmt.setInt(8, num_of_training_iterations);
        preparedStmt.setString(9, properties.getProperty("classifier"));
        preparedStmt.setInt(10, labeled_training_set_size);
        preparedStmt.setString(11, "fixed");
        preparedStmt.execute();

        conn.close();
        return exp_id + 1;
    }

    private static int setNextExperimentID(int exp_id, String arff_name, String co_trainer, String feature_selector,
                                          String value_function, String discretizer, int num_of_training_iterations, int labeled_training_set_size, Properties properties) throws Exception {
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        Date date = new Date();
        String sql = "insert into tbl_Experiments (exp_id, arff_name, start_date, co_trainer, feature_selector, value_function, discretizer, num_of_training_iterations, classifier, labeled_training_set_size,item_insertion_policy) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        PreparedStatement preparedStmt = conn.prepareStatement(sql);
        preparedStmt.setInt (1, exp_id+1);
        preparedStmt.setString (2, arff_name);
        preparedStmt.setTimestamp   (3, new java.sql.Timestamp(new java.util.Date().getTime()));
        preparedStmt.setString(4, co_trainer);
        preparedStmt.setString(5, feature_selector);
        preparedStmt.setString(6, value_function);
        preparedStmt.setString(7, discretizer);
        preparedStmt.setInt(8, num_of_training_iterations);
        preparedStmt.setString(9, properties.getProperty("classifier"));
        preparedStmt.setInt(10, labeled_training_set_size);
        preparedStmt.setString(11, "fixed");
        preparedStmt.execute();

        conn.close();

        return exp_id + 1;
    }




}
