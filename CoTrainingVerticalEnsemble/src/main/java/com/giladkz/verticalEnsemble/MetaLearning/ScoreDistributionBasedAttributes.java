package com.giladkz.verticalEnsemble.MetaLearning;

import com.datumbox.framework.common.dataobjects.FlatDataCollection;
import com.datumbox.framework.common.dataobjects.FlatDataList;
import com.datumbox.framework.common.dataobjects.TransposeDataList;
import com.datumbox.framework.core.statistics.nonparametrics.independentsamples.KolmogorovSmirnovIndependentSamples;
import com.datumbox.framework.core.statistics.nonparametrics.onesample.ShapiroWilk;
import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.StatisticsCalculations.StatisticOperations;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.TTest;

import java.io.InputStream;
import java.util.*;

import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateAverageClassificationResults;
import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.calculateMultiplicationClassificationResults;


/**
 * Created by giladkatz on 9/24/17.
 */
public class ScoreDistributionBasedAttributes {



    private double histogramItervalSize = 0.1;



    /* When we look at the previous iterations, we can look at "windows" of varying size. This list specifies the
            number of iterations back (from the n-1) in each window. The attributes for each are calculated separately */
    private List<Integer> numOfIterationsBackToAnalyze = Arrays.asList(1,3,5,10);
    private List<Double> confidenceScoreThresholds = Arrays.asList(0.5001, 0.75, 0.9, 0.95);

    HashMap<String,TreeMap<Integer, Double>> generalPartitionPercentageByScoreHistogram = new HashMap<>();

    /**
     *
     * @param unlabeledSamplesDataset
     * @param labeledSamplesDataset
     * @param currentIterationIndex
     * @param evaluationResultsPerSetAndInteration
     * @param unifiedDatasetEvaulationResults
     * @param targetClassIndex
     * @param type
     * @param properties
     * @return
     * @throws Exception
     */
    public TreeMap<Integer,AttributeInfo> getScoreDistributionBasedAttributes(Dataset unlabeledSamplesDataset, Dataset labeledSamplesDataset
            , int currentIterationIndex, TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration,
             EvaluationPerIteraion unifiedDatasetEvaulationResults, int targetClassIndex, String type, Properties properties) throws Exception {

        TreeMap<Integer,AttributeInfo> attributes = new TreeMap<>();

        /*
        We create three groups of features:
        1) Analyzes the current score distribution of the each partition, the unified set (single classifier) and the averaging/multiplication combinations
        2) For each of the cases described in section 1, compare their score distribution to past iterations
        3) Compare the score distributions for the various approaches described above (current iteration, past iterations)

        NOTE: we currently analyze only the score distributions, but we can also extract the information provided by Weka for each evaluation
         */

        /*
        * method type: reg or td
        * 1) td: for score distribution of the dataset after "selected batch" - by the batch generator
        * 2) reg: for the score distribution of the current dataset
        * */
        if(Objects.equals(type, "td")){
            setNumOfIterationsBackToAnalyze(Arrays.asList(1,2,4,6,11));
        }else{
            setNumOfIterationsBackToAnalyze(Arrays.asList(1,3,5,10));
        }

        //region Generate the averaging and multiplication score distributions for all iterations
        //TODO: this really needs to be saved in a cache with only incremental updates. For large datasets this can take a long time
        TreeMap<Integer, TreeMap<Integer,double[]>> averageingScoreDistributionsPerIteration = new TreeMap<>();
        TreeMap<Integer, TreeMap<Integer,double[]>> multiplicationScoreDistributionsPerIteration = new TreeMap<>();

        for (int i=0; i<evaluationResultsPerSetAndInteration.size();i++){
            //start by getting the EvaluationInfo object of the respective iterations for each partition
            HashMap<Integer,EvaluationInfo> partitionsIterationEvaulatioInfos = new HashMap<>();
            for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                partitionsIterationEvaulatioInfos.put(partitionIndex, evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex));
            }

            //next, calculate the averaging score of the iteration
            TreeMap<Integer,double[]> averagingScoreDistribution = calculateAverageClassificationResults(partitionsIterationEvaulatioInfos, labeledSamplesDataset.getNumOfClasses());
            averageingScoreDistributionsPerIteration.put(i, averagingScoreDistribution);
            TreeMap<Integer,double[]> multiplicationScoreDistribution = calculateMultiplicationClassificationResults(partitionsIterationEvaulatioInfos, labeledSamplesDataset.getNumOfClasses(), labeledSamplesDataset.getClassRatios(false));
            multiplicationScoreDistributionsPerIteration.put(i, multiplicationScoreDistribution);
        }
        //endregion

        //region Group 1 - Analyzes the current score distribution of the each partition, the unified set (single classifier) and the averaging/multiplication combinations
        //region Get the score distributions of the partitions
        TreeMap<Integer,AttributeInfo> currentScoreDistributionStatistics = new TreeMap<>();
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            TreeMap<Integer,AttributeInfo> generalStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                    evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions(),targetClassIndex, "partition_" + partitionIndex,properties);
            for (int pos : generalStatisticsAttributes.keySet()) {
                currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), generalStatisticsAttributes.get(pos));
            }
        }
        //endregion

        //region Next we evaluate the current score distribution of the "mixture models" - the unified model (on all features), averaging and multiplication
        //start with the unified set
        TreeMap<Integer,AttributeInfo> unifiedSetGeneralStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                unifiedDatasetEvaulationResults.getLatestEvaluationInfo().getScoreDistributions(),targetClassIndex, "unified", properties);
        for (int pos : unifiedSetGeneralStatisticsAttributes.keySet()) {
            currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), unifiedSetGeneralStatisticsAttributes.get(pos));
        }

        //the averaging meta-features
        TreeMap<Integer,AttributeInfo> generalAveragingStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                averageingScoreDistributionsPerIteration.get(averageingScoreDistributionsPerIteration.lastKey()),targetClassIndex, "averaging", properties);
        for (int pos : generalAveragingStatisticsAttributes.keySet()) {
            currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), generalAveragingStatisticsAttributes.get(pos));
        }

        //the multiplication meta-features
        TreeMap<Integer,AttributeInfo> generalMultiplicationStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
            multiplicationScoreDistributionsPerIteration.get(multiplicationScoreDistributionsPerIteration.lastKey()),targetClassIndex, "multiplication", properties);
        for (int pos : generalMultiplicationStatisticsAttributes.keySet()) {
            currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), generalMultiplicationStatisticsAttributes.get(pos));
        }
        attributes.putAll(currentScoreDistributionStatistics);

        //endregion
        //endregion

        //region Group 2 - For each of the cases described in section 1, compare their score distribution to past iterations
        TreeMap<Integer,AttributeInfo> iterationsBasedStatisticsAttributes = new TreeMap<>();


        //region first evaluate each partition separately
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            TreeMap<Integer, AttributeInfo> paritionIterationBasedAttributes =
                    calculateScoreDistributionStatisticsOverMultipleIterations(currentIterationIndex,
                            evaluationResultsPerSetAndInteration.get(partitionIndex).getAllIterationsScoreDistributions(),targetClassIndex, "partition_" + partitionIndex, properties);
            for (int pos : paritionIterationBasedAttributes.keySet()) {
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), paritionIterationBasedAttributes.get(pos));
            }
        }
        //endregion

        //region Analyze the unified set and the averaging and multiplication rsults per iteration
        //next, get the per-iteration statistics of the unified model
        TreeMap<Integer, AttributeInfo> unifiedSetIterationBasedAttributes =
                calculateScoreDistributionStatisticsOverMultipleIterations(currentIterationIndex,
                        unifiedDatasetEvaulationResults.getAllIterationsScoreDistributions(),
                        targetClassIndex, "unified", properties);
        for (int pos : unifiedSetIterationBasedAttributes.keySet()) {
            iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), unifiedSetIterationBasedAttributes.get(pos));
        }

        //now the averaging and multiplication
        TreeMap<Integer, AttributeInfo> averagingIterationBasedAttributes =
                calculateScoreDistributionStatisticsOverMultipleIterations(1,
                        averageingScoreDistributionsPerIteration,targetClassIndex, "averaging", properties);
        for (int pos : averagingIterationBasedAttributes.keySet()) {
            iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), averagingIterationBasedAttributes.get(pos));
        }

        TreeMap<Integer, AttributeInfo> multiplicationIterationBasedAttributes =
                calculateScoreDistributionStatisticsOverMultipleIterations(1,
                        multiplicationScoreDistributionsPerIteration,targetClassIndex,"multiplication", properties);
        for (int pos : multiplicationIterationBasedAttributes.keySet()) {
            iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), multiplicationIterationBasedAttributes.get(pos));
        }
        attributes.putAll(iterationsBasedStatisticsAttributes);
        //endregion
        //endregion

        //region Group 3 - Compare the score distributions for the various approaches described above (current iteration, past iterations)
        TreeMap<Integer,AttributeInfo> crossPartitionIterationsBasedStatisticsAttributes = new TreeMap<>();

        //region We now create a single data structure containing all the distributions and then
        // use a pair of loops to evaluate every pair once
        TreeMap<Integer, TreeMap<Integer, TreeMap<Integer,double[]>>> allPartitionsAndDistributions = new TreeMap<>();
        TreeMap<Integer, String> identifierPartitionsMap = new TreeMap<>();

        //insert all the "basic" paritions (this supports more then two partitions,
        // but may case a problem during the meta learning phase)
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            allPartitionsAndDistributions.put(allPartitionsAndDistributions.size()
                    , evaluationResultsPerSetAndInteration.get(partitionIndex).getAllIterationsScoreDistributions());
            identifierPartitionsMap.put(identifierPartitionsMap.size(), "partition_" + partitionIndex);
        }

        //now the unified set
        allPartitionsAndDistributions.put(allPartitionsAndDistributions.size(), unifiedDatasetEvaulationResults.getAllIterationsScoreDistributions());
        identifierPartitionsMap.put(identifierPartitionsMap.size(), "unified");

        //Finally, the averaging and multiplication
        allPartitionsAndDistributions.put(allPartitionsAndDistributions.size(), averageingScoreDistributionsPerIteration);
        identifierPartitionsMap.put(identifierPartitionsMap.size(), "averaging");
        allPartitionsAndDistributions.put(allPartitionsAndDistributions.size(), multiplicationScoreDistributionsPerIteration);
        identifierPartitionsMap.put(identifierPartitionsMap.size(), "multiplication");
        //endregion

        //now we use a pair of loops to analyze every pair of partitions once
        for (int i=0; i<allPartitionsAndDistributions.size()-1-2; i++) {
            for (int j=1; j<allPartitionsAndDistributions.size()-2; j++) {
                if (i != j) {
                    TreeMap<Integer, AttributeInfo> crossPartitionFeatures =
                            calculateScoreDistributionStatisticsOverMultipleSetsAndIterations(
                                    currentIterationIndex
                                    , allPartitionsAndDistributions.get(i),
                                    allPartitionsAndDistributions.get(j)
                                    , targetClassIndex,
                                    "_" + identifierPartitionsMap.get(i) + "_" + identifierPartitionsMap.get(j),
                                    properties);
                    for (int key: crossPartitionFeatures.keySet()) {
                        crossPartitionIterationsBasedStatisticsAttributes.put
                                (crossPartitionIterationsBasedStatisticsAttributes.size(), crossPartitionFeatures.get(key));
                    }
                }
            }
        }
        //attributes.putAll(crossPartitionIterationsBasedStatisticsAttributes);
        //endregion


        return attributes;
    }


    /**
     * Used to generate Group 3
     * @param currentIteration
     * @param iterationsEvaluationInfo1
     * @param iterationsEvaluationInfo2
     * @param targetClassIndex
     * @param properties
     * @return
     * @throws Exception
     */
    public TreeMap<Integer,AttributeInfo> calculateScoreDistributionStatisticsOverMultipleSetsAndIterations(
            int currentIteration, TreeMap<Integer, TreeMap<Integer, double[]>> iterationsEvaluationInfo1,
            TreeMap<Integer, TreeMap<Integer, double[]>> iterationsEvaluationInfo2, int targetClassIndex,
            String identifier, Properties properties) throws Exception {
        //The comparison is conducted as follows: we calculate difference statistics on
        // the first, top 5, top 10 etc. Comparisons are only performed for the same time index
        TreeMap<Integer,AttributeInfo> partitionBasedStatisticsAttributes = new TreeMap<>();

        //get confidence scores per group
        TreeMap<Integer, double[]> currentIterationScoreDistGroup1 = iterationsEvaluationInfo1.get(currentIteration);
        TreeMap<Integer, double[]> currentIterationScoreDistGroup2 = iterationsEvaluationInfo2.get(currentIteration);

        //instance delta score on target class
        double[] instanceScoreTargetClassGroup1 = new double[currentIterationScoreDistGroup1.keySet().size()];
        double[] instanceScoreTargetClassGroup2 = new double[currentIterationScoreDistGroup2.keySet().size()];
        double[] instanceDeltaScoreTargetClass = new double[currentIterationScoreDistGroup1.keySet().size()];
        DescriptiveStatistics deltaBetweenGroups= new DescriptiveStatistics();
        int instanceScoreTargetClass_cnt = 0;
        for (int ins:currentIterationScoreDistGroup1.keySet()) {
            double insScoreGroup1 = currentIterationScoreDistGroup1.get(ins)[targetClassIndex];
            double insScoreGroup2 = currentIterationScoreDistGroup2.get(ins)[targetClassIndex];
            double delta = insScoreGroup1 - insScoreGroup2;
            deltaBetweenGroups.addValue(delta);

            instanceScoreTargetClassGroup1[instanceScoreTargetClass_cnt]=insScoreGroup1;
            instanceScoreTargetClassGroup2[instanceScoreTargetClass_cnt]=insScoreGroup2;
            instanceDeltaScoreTargetClass[instanceScoreTargetClass_cnt]=delta;
            instanceScoreTargetClass_cnt++;
        }
        //extract DescriptiveStatistics statistics from instanceDeltaScoreTargetClass
        //stats for distance cross partition
        //max
        AttributeInfo maxDeltaScoreDist = new AttributeInfo
                ("maxDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, Column.columnType.Numeric
                        , deltaBetweenGroups.getMax(), -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), maxDeltaScoreDist);
        //min
        AttributeInfo minDeltaScoreDist = new AttributeInfo
                ("minDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, Column.columnType.Numeric
                        , deltaBetweenGroups.getMin(), -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), minDeltaScoreDist);
        //mean
        AttributeInfo meanDeltaScoreDist = new AttributeInfo
                ("meanDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, Column.columnType.Numeric
                        , deltaBetweenGroups.getMean(), -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), meanDeltaScoreDist);
        //std
        AttributeInfo stdDeltaScoreDist = new AttributeInfo
                ("stdDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, Column.columnType.Numeric
                        , deltaBetweenGroups.getStandardDeviation(), -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), stdDeltaScoreDist);
        //p-50
        AttributeInfo medianDeltaScoreDist = new AttributeInfo
                ("medianDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, Column.columnType.Numeric
                        , deltaBetweenGroups.getPercentile(50), -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), medianDeltaScoreDist);

        //t-test on scores per group
        TTest tTest = new TTest();
        double TTestStatistic = tTest.t(instanceScoreTargetClassGroup1,instanceScoreTargetClassGroup2);
        AttributeInfo tTest_att = new AttributeInfo("t_test_"+identifier+"_iteration_"+currentIteration
                ,Column.columnType.Numeric, TTestStatistic, -1);
        partitionBasedStatisticsAttributes.put(partitionBasedStatisticsAttributes.size(), tTest_att);


        return partitionBasedStatisticsAttributes;
    }


    /**
     * Used to generate Group 2 features. This function is used to analyze the score distributions over time FOR A SINGLE PARTITION. It is also important to remember that everything in this function is calculated with respect to the CURRENT iteration.
     * @param currentIteration
     * @param iterationsEvaluationInfo
     * @param targetClassIndex
     * @param properties
     * @return
     * @throws Exception
     */
    public TreeMap<Integer,AttributeInfo> calculateScoreDistributionStatisticsOverMultipleIterations(int currentIteration,
            TreeMap<Integer, TreeMap<Integer,double[]>> iterationsEvaluationInfo, int targetClassIndex, String identifier, Properties properties) throws Exception {

        //region Statistics on previous iterations

        //Statistics on the changes in confidence score for each instances compared to previous iterations
        TreeMap<Integer, DescriptiveStatistics> confidenceScoreDifferencesPerSingleIteration = new TreeMap<>();

        //instanceID -> num of iterations backwards -> values
        TreeMap<Integer, HashMap<Integer,DescriptiveStatistics>> confidenceScoreDifferencesPerInstance = new TreeMap<>();

        /*A histogram of the differences between the current scores histogram and one of the previous iterations' (i.e. the changes in the percentages assigned to each "box") */
        TreeMap<Integer, HashMap<Integer, Double>> generalPercentageScoresDiffHistogramByIteration = new TreeMap<>();

        //The paired T-Test values for the current iteration and one of the previous iterations
        TreeMap<Integer, Double> tTestValueForCurrentAndPreviousIterations = new TreeMap<>();

        //Statistics on the Paired T-Test values of the previous X iterations (not including the current)
        TreeMap<Integer, DescriptiveStatistics> previousIterationsTTestStatistics = new TreeMap<>();

        //Statistics on the percentage of instances that changed labels given a confidence threshold and number of iterations back
        TreeMap<Integer, HashMap<Double, Double>> labelChangePercentageByIterationAndThreshold = new TreeMap<>();
        //endregion


        TreeMap<Integer,AttributeInfo> iterationsBasedStatisticsAttributes = new TreeMap<>();

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);


        /*We operate under the assumption that the generalScoreStats object, which contains the current iteration's confidence scores
         * has already been populated */

        /* We begin by loading the deltas of the confidence scores of consecutive iterations into the object.*/
        for (int i=1; i<iterationsEvaluationInfo.size(); i++) {
            confidenceScoreDifferencesPerSingleIteration.put(i, new DescriptiveStatistics());
            TreeMap<Integer,double[]> currentIterationScoreDistribution =  iterationsEvaluationInfo.get(i);
            TreeMap<Integer,double[]> previousIterationScoreDistribution = iterationsEvaluationInfo.get(i-1);

            //Extract the per-iteration information
            for (int j : currentIterationScoreDistribution.keySet()) {
                double delta = currentIterationScoreDistribution.get(j)[targetClassIndex]-previousIterationScoreDistribution.get(j)[targetClassIndex];
                confidenceScoreDifferencesPerSingleIteration.get(i).addValue(delta);
            }

            //extract the per-instance information
            for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
                /* If the iteration is within the "scope" of the analysis (i.e. the number of iterations back falls within one or more of the ranges */
                if (currentIteration - (i-1) <= numOfIterationsBack) {
                    for (int j : currentIterationScoreDistribution.keySet()) {
                        if (!confidenceScoreDifferencesPerInstance.containsKey(j)) {
                            confidenceScoreDifferencesPerInstance.put(j, new HashMap<>());
                        }
                        if (!confidenceScoreDifferencesPerInstance.get(j).containsKey(numOfIterationsBack)) {
                            confidenceScoreDifferencesPerInstance.get(j).put(numOfIterationsBack, new DescriptiveStatistics());
                        }
                        double delta = currentIterationScoreDistribution.get(j)[targetClassIndex]-previousIterationScoreDistribution.get(j)[targetClassIndex];
                        confidenceScoreDifferencesPerInstance.get(j).get(numOfIterationsBack).addValue(delta);
                    }
                }
            }
        }

        //now we produce the statistics for a varying number of backwards iterations
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIteration >= numOfIterationsBack) {

                //region start by generating statistics on the ITERATION-LEVEL statistics
                DescriptiveStatistics iteraionTempDSMax = new DescriptiveStatistics();
                DescriptiveStatistics iterationTempDSMin = new DescriptiveStatistics();
                DescriptiveStatistics iterationTempDSAvg = new DescriptiveStatistics();
                DescriptiveStatistics iterationTempDSStdev = new DescriptiveStatistics();
                DescriptiveStatistics iterationTempDSMedian = new DescriptiveStatistics();
                for (int i = currentIteration; i > (currentIteration - numOfIterationsBack); i--) {
                    try{
                        iteraionTempDSMax.addValue(confidenceScoreDifferencesPerSingleIteration.get(i).getMax());
                        iterationTempDSMin.addValue(confidenceScoreDifferencesPerSingleIteration.get(i).getMin());
                        iterationTempDSAvg.addValue(confidenceScoreDifferencesPerSingleIteration.get(i).getMean());
                        iterationTempDSStdev.addValue(confidenceScoreDifferencesPerSingleIteration.get(i).getStandardDeviation());
                        iterationTempDSMedian.addValue(confidenceScoreDifferencesPerSingleIteration.get(i).getPercentile(50));
                    }catch (Exception e){
                        continue;
                    }

                }

                // Now we extract the AVG and Stdev of these temp statistics
                AttributeInfo maxAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, iteraionTempDSMax.getMean(), -1);
                AttributeInfo maxStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, iteraionTempDSMax.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxStdevAtt);

                AttributeInfo minAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMinDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSMin.getMean(), -1);
                AttributeInfo minStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMinDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSMin.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minStdevAtt);

                AttributeInfo avgAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSAvg.getMean(), -1);
                AttributeInfo avgStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSAvg.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgStdevAtt);

                AttributeInfo stdevAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSStdev.getMean(), -1);
                AttributeInfo stdevStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSStdev.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevStdevAtt);

                AttributeInfo medianAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSMedian.getMean(), -1);
                AttributeInfo medianStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, iterationTempDSMedian.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianStdevAtt);
                //endregion

                //region now we calculate the INSTANCE-LEVEL statistics
                DescriptiveStatistics instanceTempDSMax = new DescriptiveStatistics();
                DescriptiveStatistics instanceTempDSMin = new DescriptiveStatistics();
                DescriptiveStatistics instanceTempDSAvg = new DescriptiveStatistics();
                DescriptiveStatistics instanceTempDSStdev = new DescriptiveStatistics();
                DescriptiveStatistics instanceTempDSMedian = new DescriptiveStatistics();

                for (int instanceID : confidenceScoreDifferencesPerInstance.keySet()) {
                    try{
                        instanceTempDSMax.addValue(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack).getMax());
                        instanceTempDSMin.addValue(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack).getMin());
                        instanceTempDSAvg.addValue(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack).getMean());
                        instanceTempDSStdev.addValue(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack).getStandardDeviation());
                        instanceTempDSMedian.addValue(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack).getPercentile(50));
                    } catch (Exception e){
                        continue;
                    }

                }

                AttributeInfo maxAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMax.getMean(), -1);
                AttributeInfo maxStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMax.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxStdevPerInstanceAtt);

                AttributeInfo minAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMinDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMin.getMean(), -1);
                AttributeInfo minStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMinDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMin.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minStdevPerInstanceAtt);

                AttributeInfo avgAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSAvg.getMean(), -1);
                AttributeInfo avgStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSAvg.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgStdevPerInstanceAtt);

                AttributeInfo stdevAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSStdev.getMean(), -1);
                AttributeInfo stdevStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSStdev.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevStdevPerInstanceAtt);

                AttributeInfo medianAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMedian.getMean(), -1);
                AttributeInfo medianStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, instanceTempDSMedian.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianStdevPerInstanceAtt);
                //endregion
            }
            else {
                //if the conditions are not met, just add -1 to all relevant attributes
                //TODO: consider adding a question mark instead
                //region Add -1 instead of the values

                //region Iteration-level values
                AttributeInfo maxAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo maxStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxStdevAtt);

                AttributeInfo minAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMinDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo minStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMinDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minStdevAtt);

                AttributeInfo avgAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo avgStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgStdevAtt);

                AttributeInfo stdevAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo stdevStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevStdevAtt);

                AttributeInfo medianAvgAtt = new AttributeInfo(numOfIterationsBack + "iterationsAverageOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo medianStdevAtt = new AttributeInfo(numOfIterationsBack + "iterationsStdevOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianAvgAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianStdevAtt);
                //endregion

                //region Instance-level values
                AttributeInfo maxAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo maxStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMaxDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxStdevPerInstanceAtt);

                AttributeInfo minAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMinDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo minStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMinDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minStdevPerInstanceAtt);

                AttributeInfo avgAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo avgStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfAvgDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgStdevPerInstanceAtt);

                AttributeInfo stdevAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo stdevStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfStdevDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevStdevPerInstanceAtt);

                AttributeInfo medianAvgPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesAverageOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                AttributeInfo medianStdevPerInstanceAtt = new AttributeInfo(numOfIterationsBack + "instancesStdevOfMedianDelta" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianAvgPerInstanceAtt);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianStdevPerInstanceAtt);
                //endregion

                //endregion
            }
        }


        //now we generate the histograms at different time points and compare
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIteration >= numOfIterationsBack) {
                //region Generate the attbributes representing the histogram changes
                generalPercentageScoresDiffHistogramByIteration.put(numOfIterationsBack, new HashMap<>());

                //First, calculate the percentage for each histogram, like we did in the "general statistics" section
                for (double i=0.0; i<1; i+=histogramItervalSize) {
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).put((int)Math.round(i/histogramItervalSize),0.0);
                }

                TreeMap<Integer,double[]> earlierIterationScoreDistribution = iterationsEvaluationInfo.get(currentIteration-numOfIterationsBack);
                for (int i : earlierIterationScoreDistribution.keySet()) {
                    int histogramIndex = (int)Math.round(earlierIterationScoreDistribution.get(i)[targetClassIndex]/histogramItervalSize);
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).put(histogramIndex, generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(histogramIndex)+1.0);
                }
                for (int key : generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).keySet()) {
                    double histoCellPercentage = generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(key)/earlierIterationScoreDistribution.keySet().size();
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).put(key, histoCellPercentage);
                }

                //now, generate the attributes representing the changes in the histogram
                DescriptiveStatistics histogramStatistics = new DescriptiveStatistics();
                for (int key : generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).keySet()) {
                    double delta = generalPartitionPercentageByScoreHistogram.get(identifier).get(key) - generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(key);
                    histogramStatistics.addValue(delta);
                }
                AttributeInfo maxDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, Column.columnType.Numeric, histogramStatistics.getMax(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxDeltaHistoAtt);

                AttributeInfo minDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, Column.columnType.Numeric, histogramStatistics.getMin(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minDeltaHistoAtt);

                AttributeInfo avgDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, Column.columnType.Numeric, histogramStatistics.getMean(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgDeltaHistoAtt);

                AttributeInfo stdevDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, Column.columnType.Numeric, histogramStatistics.getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevDeltaHistoAtt);

                AttributeInfo medianDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, Column.columnType.Numeric, histogramStatistics.getPercentile(50), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianDeltaHistoAtt);
                //endregion

            }
            else {
                //region If there are not enough iterations yet, place -1 everywhere
                AttributeInfo maxDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxDeltaHistoAtt);

                AttributeInfo minDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minDeltaHistoAtt);

                AttributeInfo avgDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgDeltaHistoAtt);

                AttributeInfo stdevDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevDeltaHistoAtt);

                AttributeInfo medianDeltaHistoAtt = new AttributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianDeltaHistoAtt);
                //endregion
            }
        }


        //region Calculate the Paired T-Test statistics of the current iteration with previous iterations
        //TO DO: understand what is this structure: iterationsEvaluationInfo -- ASK Gilad
        TTest tTest = new TTest();
        TreeMap<Integer,double[]> scoreDistributions = iterationsEvaluationInfo.get(currentIteration);
        double[] currentIterationTargetClassScoreDistributions = new double[scoreDistributions.keySet().size()];


        //The use of TreeMap rather than HashMap is supposed to ensure that the indices are always read in the same (ascending) order
        int counter1 = 0;
        for (int i  : scoreDistributions.keySet()) {
            currentIterationTargetClassScoreDistributions[counter1] = scoreDistributions.get(i)[targetClassIndex];
            counter1++;
        }

        //now for each iteration we extract its values and calculate the Paired T-Test statistic
        for (int i=0; i<iterationsEvaluationInfo.size()-1; i++) {
            TreeMap<Integer,double[]> tempScoreDistributions = iterationsEvaluationInfo.get(i);
            double[] tempIterationTargetClassScoreDistributions = new double[currentIterationTargetClassScoreDistributions.length]; //CRITICAL: because of the differences in the sizes of the unlabeled sets, we always use the one of the current iteration (which is smallest and shared by all previous iterations)
            //Note: here I use the KeySet of scoreDistributions which belongs to the CURRENT iteration. This should ensure that the values are paired
            int counter2 = 0;
            for (int j  : scoreDistributions.keySet()) {
                tempIterationTargetClassScoreDistributions[counter2] = tempScoreDistributions.get(j)[targetClassIndex];
                counter2++;
            }
            double TTestStatistic = tTest.t(currentIterationTargetClassScoreDistributions,tempIterationTargetClassScoreDistributions);
            tTestValueForCurrentAndPreviousIterations.put(i,TTestStatistic);
        }

        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIteration >= numOfIterationsBack) {
                previousIterationsTTestStatistics.put(numOfIterationsBack, new DescriptiveStatistics());
                for (int i=currentIteration-numOfIterationsBack; i<currentIteration; i++) {
                    previousIterationsTTestStatistics.get(numOfIterationsBack).addValue(tTestValueForCurrentAndPreviousIterations.get(i));
                }

                //now that we obtained the statistics of all the relevant iterations, we can generate the attributes
                AttributeInfo maxTTestStatisticForScoreDistributionAtt = new AttributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, previousIterationsTTestStatistics.get(numOfIterationsBack).getMax(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxTTestStatisticForScoreDistributionAtt);

                AttributeInfo minTTestStatisticForScoreDistributionAtt = new AttributeInfo("minTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, previousIterationsTTestStatistics.get(numOfIterationsBack).getMin(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), minTTestStatisticForScoreDistributionAtt);

                AttributeInfo avgTTestStatisticForScoreDistributionAtt = new AttributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, previousIterationsTTestStatistics.get(numOfIterationsBack).getMean(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), avgTTestStatisticForScoreDistributionAtt);

                AttributeInfo stdevTTestStatisticForScoreDistributionAtt = new AttributeInfo("stdevTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, previousIterationsTTestStatistics.get(numOfIterationsBack).getStandardDeviation(), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevTTestStatisticForScoreDistributionAtt);

                AttributeInfo medianTTestStatisticForScoreDistributionAtt = new AttributeInfo("medianTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, previousIterationsTTestStatistics.get(numOfIterationsBack).getPercentile(50), -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianTTestStatisticForScoreDistributionAtt);

            }
            else {
                //region Fill with -1 values if we don't have the required iterations
                AttributeInfo maxTTestStatisticForScoreDistributionAtt = new AttributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxTTestStatisticForScoreDistributionAtt);

                AttributeInfo minTTestStatisticForScoreDistributionAtt = new AttributeInfo("minTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxTTestStatisticForScoreDistributionAtt);

                AttributeInfo avgTTestStatisticForScoreDistributionAtt = new AttributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), maxTTestStatisticForScoreDistributionAtt);

                AttributeInfo stdevTTestStatisticForScoreDistributionAtt = new AttributeInfo("stdevTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), stdevTTestStatisticForScoreDistributionAtt);

                AttributeInfo medianTTestStatisticForScoreDistributionAtt = new AttributeInfo("medianTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), medianTTestStatisticForScoreDistributionAtt);
                //endregion
            }

        }
        //endregion


        //region Now we extract the percentage of instances that switched labels from a previous iteration using various thresholds

        /* We already have the value of the current iteration from the previous section, in the currentIterationTargetClassScoreDistributions parameter.
        *  This means that we only need to extract it for the previous iterations. */
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIteration >= numOfIterationsBack) {
                TreeMap<Integer,double[]> tempScoreDistributions = iterationsEvaluationInfo.get(currentIteration-numOfIterationsBack);
                double[] tempIterationTargetClassScoreDistributions = new double[scoreDistributions.keySet().size()];

                //NOTE: here we once again use the indices of the CURRENT iteration to keep everything correlated
                int counter2 = 0;
                for (int i  : scoreDistributions.keySet()) {
                    tempIterationTargetClassScoreDistributions[counter2] = tempScoreDistributions.get(i)[targetClassIndex];
                    counter2++;

                }

                labelChangePercentageByIterationAndThreshold.put(numOfIterationsBack, new HashMap<>());
                for (double threshold : confidenceScoreThresholds) {
                    double counter = 0;
                    for (int i=0; i<currentIterationTargetClassScoreDistributions.length; i++) {
                        if ((currentIterationTargetClassScoreDistributions[i] < threshold &&
                                tempIterationTargetClassScoreDistributions[i] >= threshold) ||
                                (currentIterationTargetClassScoreDistributions[i] >= threshold &&
                                        tempIterationTargetClassScoreDistributions[i] < threshold)) {
                            counter++;
                        }
                    }
                    labelChangePercentageByIterationAndThreshold.get(numOfIterationsBack).put(threshold,counter/tempIterationTargetClassScoreDistributions.length);
                }

                //now that we have extracted the percentages, time to generate the attributes
                for (double threshold : confidenceScoreThresholds) {
                    AttributeInfo labelPercetageChangeAtt = new AttributeInfo("labelPercentageChangeFor_" + numOfIterationsBack + "IterationsBack" + "_threshold_" + threshold+ "_" + identifier, Column.columnType.Numeric, labelChangePercentageByIterationAndThreshold.get(numOfIterationsBack).get(threshold), -1);
                    iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), labelPercetageChangeAtt);
                }
            }
            else {
                //generate attributes with -1 values if we don't have sufficient iterations
                for (double threshold : confidenceScoreThresholds) {
                    AttributeInfo labelPercetageChangeAtt = new AttributeInfo("labelPercentageChangeFor_" + numOfIterationsBack + "IterationsBack" + "_threshold_" + threshold+ "_" + identifier, Column.columnType.Numeric, -1.0, -1);
                    iterationsBasedStatisticsAttributes.put(iterationsBasedStatisticsAttributes.size(), labelPercetageChangeAtt);
                }
            }
        }
        //endregion



        return iterationsBasedStatisticsAttributes;
    }


    /**
     * Used to generate Group 1 features.
     * @param unlabeledSamplesDataset
     * @param labeledSamplesDataset
     * @param scoreDistributions
     * @param targetClassIndex
     * @param properties
     * @return
     * @throws Exception
     */
    public TreeMap<Integer,AttributeInfo> calculateGeneralScoreDistributionStatistics(Dataset unlabeledSamplesDataset, Dataset labeledSamplesDataset, TreeMap<Integer,double[]> scoreDistributions,
                                                                                      int targetClassIndex, String identifier, Properties properties) throws Exception {
        //region Statistics on the current score distribution
        Random rnd = new Random(Integer.parseInt(properties.getProperty("randomSeed")));

        //Simple general statistics on the overall scores, no additional filtering
        DescriptiveStatistics generalScoreStats = new DescriptiveStatistics();

        //A histogram that contains the percentage of the overall items in each "box". The key is the lowest value of the box
        TreeMap<Integer, Double> generalPercentageByScoreHistogram = new TreeMap<>();

        //Statistics on whether the score distribution is similar to known distirbutions
        HashMap<Double,Boolean> normalDistributionGoodnessOfFitPVAlues = new HashMap<>();
        HashMap<Double,Boolean> logNormalDistributionGoodnessOfFitPVAlue = new HashMap<>();
        HashMap<Double,Boolean> uniformDistributionGoodnessOfFitPVAlue = new HashMap<>();

        /* For a given confidence score threshold, we partition the instances. Then for each of the Data's attributes, we check correlation between the two groups. We then calculate statistics.
        Correlation needs to be calculated separately for numeric and discrete attributes. This is done both jointly and separately for numeric and discrete features*/
        HashMap<Double, DescriptiveStatistics> allFeatureCorrelationStatsByThreshold = new HashMap<>();
        HashMap<Double, DescriptiveStatistics> numericFeatureCorrelationStatsByThreshold = new HashMap<>();
        HashMap<Double, DescriptiveStatistics> discreteFeatureCorrelationStatsByThreshold = new HashMap<>();

        //The "true" imbalance ratio, based on the labeled data available
        double trainingSetImbalanceRatio;

        //the imbalance ratio according to the current labeling at various thresholds
        HashMap<Double, Double> imbalanceRatioByConfidenceScoreRatio = new HashMap<>();

        //The ratio of the previous values set and the "real" imbalance ratio
        HashMap<Double, Double> imbalanceScoreRatioToTrueRatio = new HashMap<>();

        //endregion

        TreeMap<Integer,AttributeInfo> generalStatisticsAttributes = new TreeMap<>();

        for (int i : scoreDistributions.keySet()) {
            generalScoreStats.addValue(scoreDistributions.get(i)[targetClassIndex]);
        }

        //region General stats
        AttributeInfo att0 = new AttributeInfo("maxConfidenceScore" + "_" + identifier , Column.columnType.Numeric, generalScoreStats.getMax(), -1);
        AttributeInfo att1 = new AttributeInfo("minConfidenceScore" + "_" + identifier, Column.columnType.Numeric, generalScoreStats.getMin(), -1);
        AttributeInfo att2 = new AttributeInfo("avgConfidenceScore" + "_" + identifier, Column.columnType.Numeric, generalScoreStats.getMean(), -1);
        AttributeInfo att3 = new AttributeInfo("stdevConfidenceScore" + "_" + identifier, Column.columnType.Numeric, generalScoreStats.getStandardDeviation(), -1);
        AttributeInfo att4 = new AttributeInfo("medianConfidenceScore" + "_" + identifier, Column.columnType.Numeric, generalScoreStats.getPercentile(50), -1);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), att0);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), att1);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), att2);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), att3);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), att4);
        //endregion

        //region Histogram of the percentage of score distributions
        for (double i=0.0; i<1; i+=histogramItervalSize) {
            generalPercentageByScoreHistogram.put((int)Math.round(i/histogramItervalSize),0.0);
        }

        for (int i : scoreDistributions.keySet()) {
            int histogramIndex = (int)Math.round(scoreDistributions.get(i)[targetClassIndex]/histogramItervalSize);
            generalPercentageByScoreHistogram.put(histogramIndex, generalPercentageByScoreHistogram.get(histogramIndex)+1.0);
        }
        for (int key : generalPercentageByScoreHistogram.keySet()) {
            double histoCellPercentage = generalPercentageByScoreHistogram.get(key)/scoreDistributions.keySet().size();
            generalPercentageByScoreHistogram.put(key, histoCellPercentage);
            AttributeInfo histoAtt = new AttributeInfo("generalScoreDistHisto_" + key + "_" + identifier, Column.columnType.Numeric, histoCellPercentage, -1);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), histoAtt);
        }

        //finally, save this to the global objest so that it can be used in Group2 and Group3 calculations
        generalPartitionPercentageByScoreHistogram.put(identifier, generalPercentageByScoreHistogram);
        //endregion

        //region Testing goodness of fit for multiple types of distributions

        /*Some of the statistical test cannot be applied on an infinite number of samples (and we don't want to do that,
        because that would take too long. For this reason, we begin by sampling a subset of instances (if warranted)
        and use them instead*/

        HashMap<Integer,Double> samplesConfidenceScoreValues = new HashMap<>();
        Integer[] keysetPositions = scoreDistributions.keySet().toArray(new Integer[scoreDistributions.size()]);

        int desiredNumberOfSamplesInSet = 4500;
        if (scoreDistributions.keySet().size() < desiredNumberOfSamplesInSet) {
            for (int key : scoreDistributions.keySet()) {
                samplesConfidenceScoreValues.put(key, scoreDistributions.get(key)[targetClassIndex]);
            }
        }
        else {
            while (samplesConfidenceScoreValues.size() < desiredNumberOfSamplesInSet) {
                int pos = rnd.nextInt(scoreDistributions.keySet().size());
                int keyValue = keysetPositions[pos];

                if (!samplesConfidenceScoreValues.containsKey(keyValue)) {
                    samplesConfidenceScoreValues.put(pos, scoreDistributions.get(keyValue)[targetClassIndex]);
                }
            }
        }

        List<Double> pValuesList = Arrays.asList(0.01, 0.05, 0.1);


        //region Normal distribution
        //Used to test normal distribution
        //https://www.programcreek.com/java-api-examples/index.php?source_dir=datumbox-framework-master/src/main/java/com/datumbox/framework/statistics/nonparametrics/onesample/ShapiroWilk.java
        ShapiroWilk sh = new ShapiroWilk();
        FlatDataCollection fdc1 = new FlatDataCollection(Arrays.asList(samplesConfidenceScoreValues.values().toArray()));
        for (double pval : pValuesList) {
            boolean isNormallyDistributed;
            try{
                isNormallyDistributed = sh.test(fdc1, pval);
            }catch (Exception e){
                isNormallyDistributed = false;
            }
            normalDistributionGoodnessOfFitPVAlues.put(pval, isNormallyDistributed);
        }

        for (double key : normalDistributionGoodnessOfFitPVAlues.keySet()) {
            AttributeInfo normalDistributionAtt = new AttributeInfo("isNormallyDistirbutedAt" + key + "_" + identifier, Column.columnType.Discrete, normalDistributionGoodnessOfFitPVAlues.get(key) , 2);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), normalDistributionAtt);
        }
        //endregion
        //region Log-normal distribution
        //Used to test logarithmic distribution
        //In order to check whether something is lognormal, we simply need to use ln(x) on the values (taking cate of 0's) and then check for normal distribution
        HashMap<Integer,Double> logNormalSamplesConfidenceScoreValues = new HashMap<>();
        for (Integer key: samplesConfidenceScoreValues.keySet()) {
            logNormalSamplesConfidenceScoreValues.put(key, 1 + samplesConfidenceScoreValues.get(key));
        }
        FlatDataCollection fdc2 = new FlatDataCollection(Arrays.asList(logNormalSamplesConfidenceScoreValues.values().toArray()));
        for (double pval : pValuesList) {
            boolean isLogNormallyDistributed;
            try{
                isLogNormallyDistributed = sh.test(fdc2, pval);
            }catch (Exception e){
                isLogNormallyDistributed = false;
            }

            logNormalDistributionGoodnessOfFitPVAlue.put(pval, isLogNormallyDistributed);
        }

        for (double key : logNormalDistributionGoodnessOfFitPVAlue.keySet()) {
            AttributeInfo logNormalDistributionAtt = new AttributeInfo("isLogNormallyDistirbutedAt" + key + "_" + identifier, Column.columnType.Discrete, logNormalDistributionGoodnessOfFitPVAlue.get(key) , 2);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), logNormalDistributionAtt);
        }
        //endregion

        //region Uniform distribution
        //Used to test whether the scores are distributed uniformly
        //Step 1: we generate a list of values randomly sampled from a uniform [0,1] distribution
        Double[] uniformRandomVals = new Double[samplesConfidenceScoreValues.size()];
        for (int i=0; i<samplesConfidenceScoreValues.size(); i++) {
            uniformRandomVals[i] = rnd.nextDouble();
        }

        //Now we create the objects needed for the Kolmogorov-Smirnov test
        TransposeDataList transposeDataList = new TransposeDataList();
        transposeDataList.put(0, new FlatDataList(Arrays.asList(uniformRandomVals))); //the random values
        transposeDataList.put(1, new FlatDataList(Arrays.asList(samplesConfidenceScoreValues.values().toArray()))); //the values we sampled from the results

        for (double pval : pValuesList) {
            boolean isUniformlyDistributed;
            try{
                isUniformlyDistributed = KolmogorovSmirnovIndependentSamples.test(transposeDataList, true, pval);
            }catch (Exception e){
                isUniformlyDistributed = false;
            }
            uniformDistributionGoodnessOfFitPVAlue.put(pval,isUniformlyDistributed);
        }

        for (double key : uniformDistributionGoodnessOfFitPVAlue.keySet()) {
            AttributeInfo uniformDistributionAtt = new AttributeInfo("isUniformlyDistirbutedAt" + key + "_" + identifier, Column.columnType.Discrete, uniformDistributionGoodnessOfFitPVAlue.get(key) , 2);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), uniformDistributionAtt);
        }
        //endregion

        //endregion

        //region Statistical tests on the correlations of features whose instances are partitioned by confidence thresholds


        for (double threshold : confidenceScoreThresholds) {
            allFeatureCorrelationStatsByThreshold.put(threshold, new DescriptiveStatistics());
            numericFeatureCorrelationStatsByThreshold.put(threshold, new DescriptiveStatistics());
            discreteFeatureCorrelationStatsByThreshold.put(threshold, new DescriptiveStatistics());

            //Since we're going to process multiple columns, we need to get the indices of the instances which are above and below the threshold
            HashMap<Integer, Boolean> indicesOfIndicesOverTheThreshold = new HashMap<>();
            int counter = 0;
            for (int i : scoreDistributions.keySet()) {
                if (scoreDistributions.get(i)[targetClassIndex] >= threshold) {
                    indicesOfIndicesOverTheThreshold.put(counter, true);
                }
                counter++;
            }

            for (ColumnInfo ci : unlabeledSamplesDataset.getAllColumns(false)) {
                if (ci.getColumn().getType() == Column.columnType.Numeric) {
                /* We perform the T-Test for all the numeric attributes of the data.
                * It is important to note that we can't use paired-t test because the samples are not paired.*/
                    double[] belowThresholdValues = new double[scoreDistributions.keySet().size()-indicesOfIndicesOverTheThreshold.size()];
                    int belowThresholdCounter = 0;
                    double[] aboveThresholdValues = new double[indicesOfIndicesOverTheThreshold.size()];
                    int aboveThresholdCounter = 0;
                    double[] values = (double[])ci.getColumn().getValues();
                    for (int i=0; i<values.length; i++) {
                        if (indicesOfIndicesOverTheThreshold.containsKey(i)) {
                            aboveThresholdValues[aboveThresholdCounter] = values[i];
                            aboveThresholdCounter++;
                        }
                        else {
                            belowThresholdValues[belowThresholdCounter] = values[i];
                            belowThresholdCounter++;
                        }
                    }
                    if (aboveThresholdCounter > 1){
                        TTest tTest = new TTest();
                        double tTestStatistic = tTest.t(aboveThresholdValues,belowThresholdValues);
                        allFeatureCorrelationStatsByThreshold.get(threshold).addValue(tTestStatistic);
                        numericFeatureCorrelationStatsByThreshold.get(threshold).addValue(tTestStatistic);
                    }

                }
                if (ci.getColumn().getType() == Column.columnType.Discrete) {
                    int[] belowThresholdValues = new int[scoreDistributions.keySet().size()-indicesOfIndicesOverTheThreshold.size()];
                    int belowThresholdCounter = 0;
                    int[] aboveThresholdValues = new int[indicesOfIndicesOverTheThreshold.size()];
                    int aboveThresholdCounter = 0;
                    int[] values = (int[])ci.getColumn().getValues();

                    for (int i=0; i<values.length; i++) {
                        if (indicesOfIndicesOverTheThreshold.containsKey(i)) {
                            aboveThresholdValues[aboveThresholdCounter] = values[i];
                            aboveThresholdCounter++;
                        }
                        else {
                            belowThresholdValues[belowThresholdCounter] = values[i];
                            belowThresholdCounter++;
                        }
                    }

                    StatisticOperations so = new StatisticOperations();
                    int numOfDiscreteValues = ((DiscreteColumn)ci.getColumn()).getNumOfPossibleValues();
                    long[][] intersectionMatrix = so.generateChiSuareIntersectionMatrix(aboveThresholdValues, numOfDiscreteValues, belowThresholdValues, numOfDiscreteValues);
                    ChiSquareTest chiSquareTest = new ChiSquareTest();
                    double chiSquareStatistic = chiSquareTest.chiSquareTest(intersectionMatrix);
                    allFeatureCorrelationStatsByThreshold.get(threshold).addValue(chiSquareStatistic);
                    discreteFeatureCorrelationStatsByThreshold.get(threshold).addValue(chiSquareStatistic);
                }
            }
        }

        /*now that have all the test statistics we can generate the attributes for the meta model (I Could have done that in the same loop,
         * but it's more convenient to separate the two - keeps things clean */
        for (double threshold : confidenceScoreThresholds) {
            AttributeInfo allFeaturesCorrelationMaxAtt = new AttributeInfo("allAttributesCorrelationMax_" + threshold + "_" + identifier, Column.columnType.Numeric, allFeatureCorrelationStatsByThreshold.get(threshold).getMax(), -1);
            AttributeInfo allFeaturesCorrelationMinAtt = new AttributeInfo("allAttributesCorrelationMin_" + threshold + "_" + identifier, Column.columnType.Numeric, allFeatureCorrelationStatsByThreshold.get(threshold).getMin(), -1);
            AttributeInfo allFeaturesCorrelationAvgAtt = new AttributeInfo("allAttributesCorrelationAvg_" + threshold + "_" + identifier, Column.columnType.Numeric, allFeatureCorrelationStatsByThreshold.get(threshold).getMean(), -1);
            AttributeInfo allFeaturesCorrelationStdevAtt = new AttributeInfo("allAttributesCorrelationStdev_" + threshold + "_" + identifier, Column.columnType.Numeric, allFeatureCorrelationStatsByThreshold.get(threshold).getStandardDeviation(), -1);
            AttributeInfo allFeaturesCorrelationMedianAtt = new AttributeInfo("allAttributesCorrelationMedian_" + threshold + "_" + identifier, Column.columnType.Numeric, allFeatureCorrelationStatsByThreshold.get(threshold).getPercentile(50), -1);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), allFeaturesCorrelationMaxAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), allFeaturesCorrelationMinAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), allFeaturesCorrelationAvgAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), allFeaturesCorrelationStdevAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), allFeaturesCorrelationMedianAtt);

            AttributeInfo numericFeaturesCorrelationMaxAtt = new AttributeInfo("numericAttributesCorrelationMax_" + threshold + "_" + identifier, Column.columnType.Numeric, numericFeatureCorrelationStatsByThreshold.get(threshold).getMax(), -1);
            AttributeInfo numericFeaturesCorrelationMinAtt = new AttributeInfo("numericAttributesCorrelationMin_" + threshold + "_" + identifier, Column.columnType.Numeric, numericFeatureCorrelationStatsByThreshold.get(threshold).getMin(), -1);
            AttributeInfo numericFeaturesCorrelationAvgAtt = new AttributeInfo("numericAttributesCorrelationAvg_" + threshold + "_" + identifier, Column.columnType.Numeric, numericFeatureCorrelationStatsByThreshold.get(threshold).getMean(), -1);
            AttributeInfo numericFeaturesCorrelationStdevAtt = new AttributeInfo("numericAttributesCorrelationStdev_" + threshold + "_" + identifier, Column.columnType.Numeric, numericFeatureCorrelationStatsByThreshold.get(threshold).getStandardDeviation(), -1);
            AttributeInfo numericFeaturesCorrelationMedianAtt = new AttributeInfo("numericAttributesCorrelationMedian_" + threshold + "_" + identifier, Column.columnType.Numeric, numericFeatureCorrelationStatsByThreshold.get(threshold).getPercentile(50), -1);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), numericFeaturesCorrelationMaxAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), numericFeaturesCorrelationMinAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), numericFeaturesCorrelationAvgAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), numericFeaturesCorrelationStdevAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), numericFeaturesCorrelationMedianAtt);

            AttributeInfo discreteFeaturesCorrelationMaxAtt = new AttributeInfo("discreteAttributesCorrelationMax_" + threshold + "_" + identifier, Column.columnType.Numeric, discreteFeatureCorrelationStatsByThreshold.get(threshold).getMax(), -1);
            AttributeInfo discreteFeaturesCorrelationMinAtt = new AttributeInfo("discreteAttributesCorrelationMin_" + threshold + "_" + identifier, Column.columnType.Numeric, discreteFeatureCorrelationStatsByThreshold.get(threshold).getMin(), -1);
            AttributeInfo discreteFeaturesCorrelationAvgAtt = new AttributeInfo("discreteAttributesCorrelationAvg_" + threshold + "_" + identifier, Column.columnType.Numeric, discreteFeatureCorrelationStatsByThreshold.get(threshold).getMean(), -1);
            AttributeInfo discreteFeaturesCorrelationStdevAtt = new AttributeInfo("discreteAttributesCorrelationStdev_" + threshold + "_" + identifier, Column.columnType.Numeric, discreteFeatureCorrelationStatsByThreshold.get(threshold).getStandardDeviation(), -1);
            AttributeInfo discreteFeaturesCorrelationMedianAtt = new AttributeInfo("discreteAttributesCorrelationMedian_" + threshold + "_" + identifier, Column.columnType.Numeric, discreteFeatureCorrelationStatsByThreshold.get(threshold).getPercentile(50), -1);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), discreteFeaturesCorrelationMaxAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), discreteFeaturesCorrelationMinAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), discreteFeaturesCorrelationAvgAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), discreteFeaturesCorrelationStdevAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), discreteFeaturesCorrelationMedianAtt);
        }
        //endregion

        //region Imbalance ratios across different thresholds on the test set
        double numOfTargetClassInstancesInTrainingData = 0;
        double numOfNonTargetClassInstancesInTrainingData = 0;
        for (int i=0; i<labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet().length; i++)
        {
            if (targetClassIndex == i) {
                numOfTargetClassInstancesInTrainingData = labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet()[i];
            }
            else {
                numOfNonTargetClassInstancesInTrainingData += labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet()[i];
            }
        }

        trainingSetImbalanceRatio = numOfTargetClassInstancesInTrainingData / numOfNonTargetClassInstancesInTrainingData;
        AttributeInfo trainingDataImbalanceRatioAtt = new AttributeInfo("trainingDataImbalanceRatio_" + "_" + identifier, Column.columnType.Numeric, trainingSetImbalanceRatio, -1);
        generalStatisticsAttributes.put(generalStatisticsAttributes.size(), trainingDataImbalanceRatioAtt);

        for (double threshold : confidenceScoreThresholds) {
            double instancesAboveThreshold = 0;
            double instancesBelowThreshold =0;
            for (int i : scoreDistributions.keySet()) {
                if (scoreDistributions.get(i)[targetClassIndex] >= threshold) {
                    instancesAboveThreshold++;
                }
                else {
                    instancesBelowThreshold++;
                }
            }
            double imbalanceRatio = instancesAboveThreshold/instancesBelowThreshold;
            if (!Double.isNaN(imbalanceRatio) && !Double.isInfinite(imbalanceRatio)) {
                imbalanceRatioByConfidenceScoreRatio.put(threshold, imbalanceRatio);
                imbalanceScoreRatioToTrueRatio.put(threshold, imbalanceRatio/trainingSetImbalanceRatio);
            }
            else {
                imbalanceRatioByConfidenceScoreRatio.put(threshold, -1.0);
                imbalanceScoreRatioToTrueRatio.put(threshold, -1.0);
            }
            AttributeInfo testSetImbalanceRatioByThresholdAtt = new AttributeInfo("testSetImbalanceRatioByThreshold_" + "_" + identifier, Column.columnType.Numeric, imbalanceRatioByConfidenceScoreRatio.get(threshold), -1);
            AttributeInfo testAndTrainSetImbalanceRatiosAtt = new AttributeInfo("testAndTrainSetImbalanceRatios_" + "_" + identifier, Column.columnType.Numeric, imbalanceScoreRatioToTrueRatio.get(threshold), -1);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), testSetImbalanceRatioByThresholdAtt);
            generalStatisticsAttributes.put(generalStatisticsAttributes.size(), testAndTrainSetImbalanceRatiosAtt);
        }


        //endregion

        return generalStatisticsAttributes;
    }

    public void setNumOfIterationsBackToAnalyze(List<Integer> numOfIterationsBackToAnalyze) {
        this.numOfIterationsBackToAnalyze = numOfIterationsBackToAnalyze;
    }
}
