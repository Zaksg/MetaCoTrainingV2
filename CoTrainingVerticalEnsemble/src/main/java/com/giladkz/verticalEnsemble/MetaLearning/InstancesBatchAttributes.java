package com.giladkz.verticalEnsemble.MetaLearning;

import com.giladkz.verticalEnsemble.Data.AttributeInfo;
import com.giladkz.verticalEnsemble.Data.Column;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Data.EvaluationPerIteraion;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.inference.ChiSquareTest;


import java.util.*;

/**
 * Created by giladkatz on 5/9/18.
 */
public class InstancesBatchAttributes {

    private List<Integer> numOfIterationsBackToAnalyze = Arrays.asList(1,3,5,10);
    private List<Double> confidenceScoreThresholds = Arrays.asList(0.5001, 0.75, 0.9, 0.95);

    /**
     *
     * @param trainingDataset
     * @param testDataset
     * @param currentIterationIndex
     * @param evaluationResultsPerSetAndInteration
     * @param unifiedDatasetEvaulationResults
     * @param targetClassIndex
     * @param instancesBatchPos - arrayList of all instances positions
     * @param assignedLabels - structure: <instancePos, assignedLabel>
     * @param properties
     * @return
     */
    public TreeMap<Integer, AttributeInfo> getInstancesBatchAssignmentMetaFeatures(
                Dataset trainingDataset, Dataset testDataset, int currentIterationIndex
                , TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
                , EvaluationPerIteraion unifiedDatasetEvaulationResults, int targetClassIndex
                , ArrayList<Integer> instancesBatchPos, HashMap<Integer, Integer> assignedLabels, Properties properties) {


        TreeMap<Integer, AttributeInfo> instanceAttributesToReturn = new TreeMap<>();

        //batch size
        int batchSize = instancesBatchPos.size();
        AttributeInfo batchSizeAttr = new AttributeInfo
                ("batchSize", Column.columnType.Numeric, batchSize, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchSizeAttr);

        //instances labeled as 0
        int countLabel0 = 0;
        for (Integer instancePos: assignedLabels.keySet()) {
            if (assignedLabels.get(instancePos) == 0){
                countLabel0++;
            }
        }
        AttributeInfo batchLabel0 = new AttributeInfo
                ("batchLabel0", Column.columnType.Numeric, countLabel0, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchLabel0);

        //batch score distribution
        //per partition
        DescriptiveStatistics batchScoreDistLabel0 = new DescriptiveStatistics();
        DescriptiveStatistics batchScoreDistLabel1 = new DescriptiveStatistics();
        HashMap<Integer, HashMap<Integer, Double>> scorePerInstancePerPartition = new HashMap<>(); //partition -> instancePos -> target class
        HashMap<Integer, HashMap<Integer, double[]>> distanceBatchPairsPerPartition = new HashMap<>(); //partition -> instancePos -> both class
        for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            //collect scores for statistics
            DescriptiveStatistics batchScoreDistPerPartition = new DescriptiveStatistics();
            HashMap<Integer, Double> scorePerInstanceTemp = new HashMap<>();
            HashMap<Integer, double[]> scorePerInstanceBothClasses = new HashMap<>();
            for (Integer instancePos: assignedLabels.keySet()) {
                TreeMap<Integer,double[]> instanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                batchScoreDistPerPartition.addValue(instanceScore.get(instancePos)[targetClassIndex]);
                scorePerInstanceTemp.put(instancePos, instanceScore.get(instancePos)[targetClassIndex]);
                scorePerInstanceBothClasses.put(instancePos, instanceScore.get(instancePos));
                //collect scores for statistics per label
                if (assignedLabels.get(instancePos) == 0){
                    batchScoreDistLabel0.addValue(evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(instancePos)[assignedLabels.get(instancePos)]);
                }
                else if (assignedLabels.get(instancePos) == 1){
                    batchScoreDistLabel1.addValue(evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(instancePos)[assignedLabels.get(instancePos)]);
                }
            }
            scorePerInstancePerPartition.put(partitionIndex, scorePerInstanceTemp);
            distanceBatchPairsPerPartition.put(partitionIndex, scorePerInstanceBothClasses);
            //max
            AttributeInfo batchScoreMax = new AttributeInfo
                    ("batchScoreMax_" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMax(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMax);
            //min
            AttributeInfo batchScoreMin = new AttributeInfo
                    ("batchScoreMin_" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMin(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMin);
            //mean
            AttributeInfo batchScoreMean = new AttributeInfo
                    ("batchScoreMean_" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMean(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMean);
            //std
            AttributeInfo batchScoreStd = new AttributeInfo
                    ("batchScoreStd_" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getStandardDeviation(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStd);
            //p-50
            AttributeInfo batchScoreMedian = new AttributeInfo
                    ("batchScoreMedian_" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getPercentile(50), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedian);

            /*pair distance
            double[] distancePerPair = new double[instanceAttributesToReturn.size()];
            int distancePerPairCounter = 0;
            for (int i = 0; i <  batchScoreDistPerPartition.getValues().length - 1; i++) {
                for (int j = i+1; j < batchScoreDistPerPartition.getValues().length; j++) {
                    distancePerPair[distancePerPairCounter] = Math.abs(batchScoreDistPerPartition.getValues()[i] - batchScoreDistPerPartition.getValues()[j]);
                    distancePerPairCounter++;
                }
            }
            distanceBatchPairsPerPartition.put(partitionIndex, distancePerPair);*/
        }

        //statistics per labels
        //label 0
        //max
        AttributeInfo batchScoreMaxLabel0 = new AttributeInfo
                ("batchScoreMaxLabel_0", Column.columnType.Numeric, batchScoreDistLabel0.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMaxLabel0);
        //min
        AttributeInfo batchScoreMinLabel0 = new AttributeInfo
                ("batchScoreMinLabel_0", Column.columnType.Numeric, batchScoreDistLabel0.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMinLabel0);
        //mean
        AttributeInfo batchScoreMeanLabel0 = new AttributeInfo
                ("batchScoreMeanLabel_0", Column.columnType.Numeric, batchScoreDistLabel0.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMeanLabel0);
        //std
        AttributeInfo batchScoreStdLabel0 = new AttributeInfo
                ("batchScoreStdLabel_0", Column.columnType.Numeric, batchScoreDistLabel0.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStdLabel0);
        //p-50
        AttributeInfo batchScoreMedianLabel0 = new AttributeInfo
                ("batchScoreMedianLabel_0", Column.columnType.Numeric, batchScoreDistLabel0.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedianLabel0);
        //label 1
        //max
        AttributeInfo batchScoreMaxLabel1 = new AttributeInfo
                ("batchScoreMaxLabel_1", Column.columnType.Numeric, batchScoreDistLabel1.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMaxLabel1);
        //min
        AttributeInfo batchScoreMinLabel1 = new AttributeInfo
                ("batchScoreMinLabel_1", Column.columnType.Numeric, batchScoreDistLabel1.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMinLabel1);
        //mean
        AttributeInfo batchScoreMeanLabel1 = new AttributeInfo
                ("batchScoreMeanLabel_1", Column.columnType.Numeric, batchScoreDistLabel1.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMeanLabel1);
        //std
        AttributeInfo batchScoreStdLabel1 = new AttributeInfo
                ("batchScoreStdLabel_1", Column.columnType.Numeric, batchScoreDistLabel1.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStdLabel1);
        //p-50
        AttributeInfo batchScoreMedianLabel1 = new AttributeInfo
                ("batchScoreMedianLabel_1", Column.columnType.Numeric, batchScoreDistLabel1.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedianLabel1);

        //delta between partition - per instance
        DescriptiveStatistics batchDeltaScore = new DescriptiveStatistics();
        for (Integer instancePos: scorePerInstancePerPartition.get(0).keySet()) {
            double instanceDelta = Math.abs(scorePerInstancePerPartition.get(0).get(instancePos)
                    - scorePerInstancePerPartition.get(1).get(instancePos));
            batchDeltaScore.addValue(instanceDelta);
        }
        //max
        AttributeInfo batchDeltaScoreMax = new AttributeInfo
                ("batchDeltaScoreMax", Column.columnType.Numeric, batchDeltaScore.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMax);
        //min
        AttributeInfo batchDeltaScoreMin = new AttributeInfo
                ("batchDeltaScoreMin", Column.columnType.Numeric, batchDeltaScore.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMin);
        //mean
        AttributeInfo batchDeltaScoreMean = new AttributeInfo
                ("batchDeltaScoreMean", Column.columnType.Numeric, batchDeltaScore.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMean);
        //std
        AttributeInfo batchDeltaScoreStd = new AttributeInfo
                ("batchDeltaScoreStd", Column.columnType.Numeric, batchDeltaScore.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreStd);
        //p-50
        AttributeInfo batchDeltaScoreMedian = new AttributeInfo
                ("batchDeltaScoreMedian", Column.columnType.Numeric, batchDeltaScore.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMedian);

        //distance statistics
        //random sampling - 1000 instances
        Random rnd = new Random(Integer.parseInt(properties.getProperty("randomSeed")));
        HashMap<Integer, HashMap<Integer, double[]>> randomSampleScores = new HashMap<>(); //instancePos -> partition -> both class
        HashMap<Integer, DescriptiveStatistics[]> axisStatsPerPartitionMap = new HashMap<>(); //partition -> stats per class
        HashMap<Integer, DescriptiveStatistics> distancePerPartitionMap = new HashMap<>(); //partition -> euclidean distance
        for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            axisStatsPerPartitionMap.put(partitionIndex, new DescriptiveStatistics[2]); //2 = number of classes
            distancePerPartitionMap.put(partitionIndex, new DescriptiveStatistics());
        }
        int randSizeSample = Math.min((int)Math.round(trainingDataset.getNumberOfRows()*0.5),1000);
        //generate random sampling and store by instance, partition and scores
        while (randomSampleScores.size() < randSizeSample){
            int rndInstancePosition = rnd.nextInt(Math.min((int)Math.round(trainingDataset.getNumberOfRows()),1000));
            if (!randomSampleScores.containsKey(rndInstancePosition)) {
                HashMap<Integer, double[]> instancePartitionscoreTemp = new HashMap<>();
                for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    Integer keyByPosition = (Integer) evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().keySet().toArray()[rndInstancePosition];
                    double[] instanceScoreDistRnd = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(keyByPosition);
                    instancePartitionscoreTemp.put(partitionIndex,instanceScoreDistRnd);

                    //add classes to statistics calcs
                    for (int i = 0; i < instanceScoreDistRnd.length; i++) {
                        DescriptiveStatistics temp = new DescriptiveStatistics();
                        temp.addValue(instanceScoreDistRnd[i]);
                        axisStatsPerPartitionMap.get(partitionIndex)[i] = temp;
                    }
                }
                randomSampleScores.put(rndInstancePosition, instancePartitionscoreTemp);
            }
        }
        //calculating avg per partition and axis
        DescriptiveStatistics distanceBatchCrossPartition = new DescriptiveStatistics();
        for (Integer batchInstancePos: assignedLabels.keySet()) {
            for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                double euclideanDistBatch = 0.0;
                int numOfClasses = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(batchInstancePos).length;
                for (int i = 0; i < numOfClasses; i++) {
                    double batchInstanceScoreDistClassI = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(batchInstancePos)[i];
                    double rndSampleAvgClassI = axisStatsPerPartitionMap.get(partitionIndex)[i].getMean();
                    euclideanDistBatch += Math.pow(batchInstanceScoreDistClassI - rndSampleAvgClassI, 2);
                }
                euclideanDistBatch = Math.sqrt(euclideanDistBatch);
                AttributeInfo euclideanDistBatchAttr = new AttributeInfo
                        ("batchDistanceFromAvgPartition_"+partitionIndex+"_instance_" + batchInstancePos, Column.columnType.Numeric, euclideanDistBatch, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), euclideanDistBatchAttr);
                distancePerPartitionMap.get(partitionIndex).addValue(euclideanDistBatch);
                distanceBatchCrossPartition.addValue(euclideanDistBatch);
            }
        }
        //statistics for batch distances
        for (int i = 0; i < distancePerPartitionMap.size() ; i++) {
            //max
            AttributeInfo batchDistancePerPartitionMax = new AttributeInfo
                    ("batchDistancePerPartitionMax_"+i, Column.columnType.Numeric, distancePerPartitionMap.get(i).getMax(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancePerPartitionMax);
            //min
            AttributeInfo batchDistancePerPartitionMin = new AttributeInfo
                    ("batchDistancePerPartitionMin_"+i, Column.columnType.Numeric, distancePerPartitionMap.get(i).getMin(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancePerPartitionMin);
            //mean
            AttributeInfo batchDistancePerPartitionMean = new AttributeInfo
                    ("batchDistancePerPartitionMean_"+i, Column.columnType.Numeric, distancePerPartitionMap.get(i).getMean(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancePerPartitionMean);
            //std
            AttributeInfo batchDistancePerPartitionStd = new AttributeInfo
                    ("batchDistancePerPartitionStd_"+i, Column.columnType.Numeric, distancePerPartitionMap.get(i).getStandardDeviation(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancePerPartitionStd);
            //p-50
            AttributeInfo batchDistancePerPartitionMedian = new AttributeInfo
                    ("batchDistancePerPartitionMedian_"+i, Column.columnType.Numeric, distancePerPartitionMap.get(i).getPercentile(50), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancePerPartitionMedian);
        }
        //stats for distance cross partition
        //max
        AttributeInfo batchDistanceCrossPartitionMax = new AttributeInfo
                ("batchDistanceCrossPartitionMax", Column.columnType.Numeric, distanceBatchCrossPartition.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistanceCrossPartitionMax);
        //min
        AttributeInfo batchDistanceCrossPartitionMin = new AttributeInfo
                ("batchDistanceCrossPartitionMin", Column.columnType.Numeric, distanceBatchCrossPartition.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistanceCrossPartitionMin);
        //mean
        AttributeInfo batchDistanceCrossPartitionMean = new AttributeInfo
                ("batchDistanceCrossPartitionMean", Column.columnType.Numeric, distanceBatchCrossPartition.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistanceCrossPartitionMean);
        //std
        AttributeInfo batchDistancecrossPartitionStd = new AttributeInfo
                ("batchDistanceCrossPartitionStd", Column.columnType.Numeric, distanceBatchCrossPartition.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistancecrossPartitionStd);
        //p-50
        AttributeInfo batchDistanceCrossPartitionMedian = new AttributeInfo
                ("batchDistanceCrossPartitionMedian", Column.columnType.Numeric, distanceBatchCrossPartition.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDistanceCrossPartitionMedian);

        //paired t-test
        TTest batchTtest;
        ChiSquareTest batchChiSquareTest;
        DescriptiveStatistics batchPartitionIterationBackTtestScore = new DescriptiveStatistics();
        DescriptiveStatistics batchIterationBackTtestScore = new DescriptiveStatistics();
        for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
            TreeMap<Integer,double[]> currentIterScoreDist = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
            double[] currentIterationTargetClassScoreDistribution = new double[currentIterScoreDist.keySet().size()];
            int currentIterationTargetClassScoreDistribution_cnt = 0;
            for (int i : currentIterScoreDist.keySet()) {
                currentIterationTargetClassScoreDistribution[currentIterationTargetClassScoreDistribution_cnt] =
                        currentIterScoreDist.get(i)[targetClassIndex];
                currentIterationTargetClassScoreDistribution_cnt++;
            }
            //consider: implement a paired T-test by calculating for the same indices
            for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
                if (currentIterationIndex >= numOfIterationsBack) {
                    TreeMap<Integer,double[]> prevIterScoreDist = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex - numOfIterationsBack).getScoreDistributions();
                    double[] prevIterationTargetClassScoreDistribution = new double[prevIterScoreDist.keySet().size()];
                    int prevIterationTargetClassScoreDistribution_cnt = 0;
                    for (int j : currentIterScoreDist.keySet()) {
                        prevIterationTargetClassScoreDistribution[prevIterationTargetClassScoreDistribution_cnt] =
                                prevIterScoreDist.get(j)[targetClassIndex];
                        prevIterationTargetClassScoreDistribution_cnt++;
                    }
                    //t-test
                    batchTtest = new TTest();
                    double batchTTestStatistic = batchTtest.t(currentIterationTargetClassScoreDistribution, prevIterationTargetClassScoreDistribution);
                    batchPartitionIterationBackTtestScore.addValue(batchTTestStatistic);
                    batchIterationBackTtestScore.addValue(batchTTestStatistic);
                    //insert t-test to the attributes
                    AttributeInfo batchTTestStatisticAttr = new AttributeInfo
                            ("batchTtestScoreOnPartition_"+partitionIndex+"_iterationBack_"+numOfIterationsBack, Column.columnType.Numeric, batchTTestStatistic, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTTestStatisticAttr);
                    //chi square

                    batchChiSquareTest = new ChiSquareTest();
                }
                else{
                    AttributeInfo batchTTestStatisticAttr = new AttributeInfo
                            ("batchTtestScoreOnPartition_"+partitionIndex+"_iterationBack_"+numOfIterationsBack, Column.columnType.Numeric, -1.0, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTTestStatisticAttr);
                }
            }
            //max
            AttributeInfo batchPartitionTtestScoreMax = new AttributeInfo
                    ("batchPartition_" +partitionIndex+"_TtestScoreMax", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMax(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMax);
            //min
            AttributeInfo batchPartitionTtestScoreMin = new AttributeInfo
                    ("batchPartition_" +partitionIndex+"_TtestScoreMin", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMin(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMin);
            //mean
            AttributeInfo batchPartitionTtestScoreMean = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreMean", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMean(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMean);
            //std
            AttributeInfo batchPartitionTtestScoreStd = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreStd", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getStandardDeviation(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreStd);
            //p-50
            AttributeInfo batchPartitionTtestScoreMedian = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreMedian", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getPercentile(50), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMedian);
        }
        //statistics on T-test scores for all batch
        //max
        AttributeInfo batchTtestScoreMax = new AttributeInfo
                ("batchTtestScoreMax", Column.columnType.Numeric, batchIterationBackTtestScore.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMax);
        //min
        AttributeInfo batchTtestScoreMin = new AttributeInfo
                ("batchTtestScoreMin", Column.columnType.Numeric, batchIterationBackTtestScore.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMin);
        //mean
        AttributeInfo batchTtestScoreMean = new AttributeInfo
                ("batchTtestScoreMean", Column.columnType.Numeric, batchIterationBackTtestScore.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMean);
        //std
        AttributeInfo batchTtestScoreStd = new AttributeInfo
                ("batchTtestScoreStd", Column.columnType.Numeric, batchIterationBackTtestScore.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreStd);
        //p-50
        AttributeInfo batchTtestScoreMedian = new AttributeInfo
                ("batchTtestScoreMedian", Column.columnType.Numeric, batchIterationBackTtestScore.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMedian);

        //fix NaN: convert to -1.0
        for (Map.Entry<Integer,AttributeInfo> entry : instanceAttributesToReturn.entrySet()){
            AttributeInfo ai = entry.getValue();
            if (ai.getAttributeType() == Column.columnType.Numeric){
                if (ai.getValue() instanceof Double){
                    double aiVal = (double) ai.getValue();
                    if (Double.isNaN(aiVal)){
                        ai.setValue(-1.0);
                    }
                }
                else if (ai.getValue() instanceof Integer){
                    Double aiVal = new Double((int)ai.getValue());
                    if (Double.isNaN(aiVal)){
                        ai.setValue(-1.0);
                    }
                }

            }
        }
        return instanceAttributesToReturn;
    }
}
