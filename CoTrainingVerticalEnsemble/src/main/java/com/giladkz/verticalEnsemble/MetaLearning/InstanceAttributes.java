package com.giladkz.verticalEnsemble.MetaLearning;

import com.giladkz.verticalEnsemble.Data.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.counting;
import static java.util.function.Function.identity;

import java.util.*;

import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.*;


public class InstanceAttributes {
    private List<Integer> numOfIterationsBackToAnalyze = Arrays.asList(1,3,5,10);

    //ToDo: change attribute names with the instance number
    //ToDo: failed insert many attributes. check if the csv helps
    public TreeMap<Integer, AttributeInfo> getInstanceAssignmentMetaFeatures
            (Dataset unlabeledSet, Dataset originalDataset, int currentIterationIndex,
             TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration,
             EvaluationPerIteraion unifiedDatasetEvaulationResults, int targetClassIndex,
             int originalIndex,int assignedLabel, Properties properties) {

        TreeMap<Integer, AttributeInfo> instanceAttributesToReturn = new TreeMap<>();

        //insert label
        AttributeInfo assignedLabelAtt = new AttributeInfo
                ("assignedLabel", Column.columnType.Discrete, assignedLabel, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), assignedLabelAtt);

        //score by partition
        //HashMap<Integer, Double> partitionsInstanceEvaulatioInfos = new HashMap<>();
        //important: partitionIndex is for the 2 classifier or more?
        HashMap<Integer, TreeMap<Integer,double[]>> iterationScoreDistPerPartition = new HashMap<>();
        double instanceSumScore = 0;
        double instanceScoreDelta = 0;
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            double instanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(originalIndex)[targetClassIndex];
            //partitionsInstanceEvaulatioInfos.put(partitionIndex, instanceScore);

            //insert score per partition
            AttributeInfo scoreClassifier = new AttributeInfo
                    ("scoreByClassifier" + partitionIndex, Column.columnType.Numeric, instanceScore, -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), scoreClassifier);

            //preparation for next calcs
            iterationScoreDistPerPartition.put(partitionIndex, evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
            instanceSumScore += instanceScore;
            if (partitionIndex%2==1){
                instanceScoreDelta += instanceScore;
            }
            else{
                instanceScoreDelta -= instanceScore;
            }
        }

        //insert the AVG score of the instance
        double instanceAvgScore = (instanceSumScore)/(evaluationResultsPerSetAndInteration.keySet().size());
        AttributeInfo instanceAvgAtt = new AttributeInfo
                ("instanceAVGScore", Column.columnType.Numeric, instanceAvgScore, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(),instanceAvgAtt);

        //insert the delta between the two classifiers
        double instanceClassifiersDeltaScore= Math.abs(instanceScoreDelta);
        AttributeInfo instanceClassifiersDeltaAtt = new AttributeInfo
                ("instanceClassifiersDeltaScore", Column.columnType.Numeric, instanceClassifiersDeltaScore, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaAtt);

        //percentile calculations
        double[] percentilePerClassPerInstancePerPartition = calculatePercentileClassificationResults(iterationScoreDistPerPartition, targetClassIndex, originalIndex);
        double instanceDeltaPercentile = 0;
        for (int partitionIndex : iterationScoreDistPerPartition.keySet()){
           double instancePercentile = percentilePerClassPerInstancePerPartition[partitionIndex];
           //insert percentile per class
            AttributeInfo percentileByClassifier = new AttributeInfo
                    ("instancePercentileByClassifier" + partitionIndex, Column.columnType.Numeric, instancePercentile, -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentileByClassifier);
            if (partitionIndex%2==1){
                instanceDeltaPercentile += instancePercentile;
            }
            else{
                instanceDeltaPercentile -= instancePercentile;
            }
        }
        //insert percentile delta
        double instanceClassifiersDeltaPercentile= Math.abs(instanceDeltaPercentile);
        AttributeInfo instanceClassifiersDeltaPercentileAtt = new AttributeInfo("instanceClassifiersDeltaScore", Column.columnType.Numeric, instanceClassifiersDeltaPercentile, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAtt);


        //iterations back
        HashMap<Integer, DescriptiveStatistics> instanceScoreByPartitionAndIteration = new HashMap<>(); //partition --> scores of all iterations
        HashMap<Integer, DescriptiveStatistics> instanceDeltaScoreByPartitionAndIteration = new HashMap<>(); // partition-> delta scores from curr iter


        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            instanceScoreByPartitionAndIteration.put(partitionIndex, new DescriptiveStatistics());
            instanceDeltaScoreByPartitionAndIteration.put(partitionIndex, new DescriptiveStatistics());

            for (Integer numOfIterationsBack : numOfIterationsBackToAnalyze) {
                //need to cut it - if no iteration: put -1 (or '?') and not skip the iteration
                if (currentIterationIndex-numOfIterationsBack>=0){
                    double instanceCurrIterationScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(originalIndex)[targetClassIndex];
                    double prevInstanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex-numOfIterationsBack).getScoreDistributions().get(originalIndex)[targetClassIndex];

                    instanceScoreByPartitionAndIteration.get(partitionIndex).addValue(prevInstanceScore);
                    //insert previous score
                    AttributeInfo instancePrevScore = new AttributeInfo
                            ("instancePrev_" + numOfIterationsBack + "_IterationsScoreClassifier" + partitionIndex, Column.columnType.Numeric, prevInstanceScore, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevScore);

                    //insert previous score - delta
                    double prevDeltaScore = Math.abs(instanceCurrIterationScore-prevInstanceScore);
                    instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).addValue(prevDeltaScore);
                    AttributeInfo instancePrevDeltaScore = new AttributeInfo
                            ("instancePrev_" + numOfIterationsBack + "_IterationsDeltaScoreClassifier_" + partitionIndex, Column.columnType.Numeric, prevDeltaScore, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevDeltaScore);
                }else{
                    AttributeInfo instancePrevDeltaScore = new AttributeInfo
                            ("instancePrev_" + numOfIterationsBack + "_IterationsDeltaScoreClassifier_" + partitionIndex, Column.columnType.Numeric, -1.0, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevDeltaScore);
                }
            }
//            if (currentIterationIndex-numOfIterationsBack>=0) {
                //stats on the iterations scores
                //max
                AttributeInfo instancePrevIterationMax = new AttributeInfo
                        ("instancePrevIterationMax_" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMax(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMax);
                //min
                AttributeInfo instancePrevIterationMin = new AttributeInfo
                        ("instancePrevIterationMin_" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMin(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMin);
                //mean
                AttributeInfo instancePrevIterationMean = new AttributeInfo
                        ("instancePrevIterationMean_" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMean(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMean);
                //std
                AttributeInfo instancePrevIterationStd = new AttributeInfo
                        ("instancePrevIterationStd_" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getStandardDeviation(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationStd);
                //p-50
                AttributeInfo instancePrevIterationMedian = new AttributeInfo
                        ("instancePrevIterationMedian_" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getPercentile(50), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMedian);

                //stats on the iterations delta scores
                //max
                AttributeInfo instanceDeltaPrevIterationMax = new AttributeInfo
                        ("instanceDeltaPrevIterationMax_" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMax(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMax);
                //min
                AttributeInfo instanceDeltaPrevIterationMin = new AttributeInfo
                        ("instanceDeltaPrevIterationMin_" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMin(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMin);
                //mean
                AttributeInfo instanceDeltaPrevIterationMean = new AttributeInfo
                        ("instanceDeltaPrevIterationMean_" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMean(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMean);
                //std
                AttributeInfo instanceDeltaPrevIterationStd = new AttributeInfo
                        ("instanceDeltaPrevIterationStd_" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getStandardDeviation(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationStd);
                //p-50
                AttributeInfo instanceDeltaPrevIterationMedian = new AttributeInfo
                        ("instanceDeltaPrevIterationMedian_" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getPercentile(50), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMedian);
//            }
        }

        //collect stats on the inner delta per iteration
        DescriptiveStatistics innerIterationDelta = new DescriptiveStatistics();
        for (int iterationBackInd = 0; iterationBackInd < instanceDeltaScoreByPartitionAndIteration.get(0).getValues().length; iterationBackInd++) {
            innerIterationDelta.addValue(Math.abs(instanceDeltaScoreByPartitionAndIteration.get(0).getValues()[iterationBackInd] - instanceDeltaScoreByPartitionAndIteration.get(1).getValues()[iterationBackInd]));
        }
        //stats on the inner iterations delta scores
        //max
        AttributeInfo instanceDeltaPrevInnerIterationMax = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMax", Column.columnType.Numeric, innerIterationDelta.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMax);
        //min
        AttributeInfo instanceDeltaPrevInnerIterationMin = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMin", Column.columnType.Numeric, innerIterationDelta.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMin);
        //mean
        AttributeInfo instanceDeltaPrevInnerIterationMean = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMean", Column.columnType.Numeric, innerIterationDelta.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMean);
        //std
        AttributeInfo instanceDeltaPrevInnerIterationStd = new AttributeInfo
                ("instanceDeltaPrevInnerIterationStd", Column.columnType.Numeric, innerIterationDelta.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationStd);
        //p-50
        AttributeInfo instanceDeltaPrevInnerIterationMedian = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMedian", Column.columnType.Numeric, innerIterationDelta.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMedian);

        //percentile on all previous iterations score
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            HashMap<Integer, TreeMap<Integer,double[]>> prevIterationScoreDistPerPartition = new HashMap<>();
            if (currentIterationIndex >= numOfIterationsBack) {
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    prevIterationScoreDistPerPartition.put(partitionIndex, evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex - numOfIterationsBack).getScoreDistributions());
                }
                double[] percentilePerClassPerInstancePerPartitionPrevIteration = calculatePercentileClassificationResults(prevIterationScoreDistPerPartition, targetClassIndex, originalIndex);
                double instancePrevIterationDeltaPercentile = 0;
                for (int partitionIndex : prevIterationScoreDistPerPartition.keySet()) {
                    double instancePercentile = percentilePerClassPerInstancePerPartitionPrevIteration[partitionIndex];
                    //insert percentile per class
                    AttributeInfo percentilePrevIteratinByClassifier = new AttributeInfo
                            ("instancePrevIteration_" + numOfIterationsBack + "_PercentileByClassifier_" + partitionIndex, Column.columnType.Numeric, instancePercentile, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentilePrevIteratinByClassifier);
                    if (partitionIndex % 2 == 1) {
                        instancePrevIterationDeltaPercentile += instancePercentile;
                    } else {
                        instancePrevIterationDeltaPercentile -= instancePercentile;
                    }
                }
                //insert percentile delta
                double instanceClassifiersDeltaPercentilePrevIteration = Math.abs(instancePrevIterationDeltaPercentile);
                AttributeInfo instanceClassifiersDeltaPercentileAttPrevIteration = new AttributeInfo("instanceClassifiersDeltaScore_" + numOfIterationsBack, Column.columnType.Numeric, instanceClassifiersDeltaPercentilePrevIteration, originalDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAttPrevIteration);
            }
            //no iteration back - insert -1 in the value
            else {
                for (int partitionIndex : prevIterationScoreDistPerPartition.keySet()) {
                    AttributeInfo percentilePrevIteratinByClassifierNoIterationBack = new AttributeInfo
                            ("percentilePrevIteratinByClassifierNoIterationBack_" + partitionIndex, Column.columnType.Numeric, -1.0, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentilePrevIteratinByClassifierNoIterationBack);
                }
                AttributeInfo instanceClassifiersDeltaPercentileAttNoIterationBack = new AttributeInfo("instanceClassifiersDeltaPercentileAttPrevIterationNoIterationBack", Column.columnType.Numeric, -1.0, originalDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAttNoIterationBack);
            }
        }

        //data from the unified dataset - no deltas

        //score
        double instanceScoreUnifiedDataset = unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions().get(originalIndex)[targetClassIndex];
        AttributeInfo instanceScoreUnifiedDatasetAtt = new AttributeInfo("instanceScoreUnifiedDataset", Column.columnType.Numeric, instanceScoreUnifiedDataset, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetAtt);
        //percentile
        HashMap<Integer, TreeMap<Integer,double[]>> instancePercentileUnifiedDatasetMap = new HashMap<>();
        instancePercentileUnifiedDatasetMap.put(0,unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
        double instancePercentileUnifiedDataset = calculatePercentileClassificationResults(instancePercentileUnifiedDatasetMap, targetClassIndex, originalIndex)[0];
        AttributeInfo instancePercentileUnifiedDatasetAtt = new AttributeInfo("instancePercentileUnifiedDataset", Column.columnType.Numeric, instancePercentileUnifiedDataset, originalDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentileUnifiedDatasetAtt);
        //iterations back
        DescriptiveStatistics scoresPerIterationUnifiedDataset = new DescriptiveStatistics();
        DescriptiveStatistics percentilePerIterationUnifiedDataset = new DescriptiveStatistics();
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIterationIndex >= numOfIterationsBack) {
                //score
                double scorePerIterationBack = unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex - numOfIterationsBack).getScoreDistributions().get(originalIndex)[targetClassIndex];
                scoresPerIterationUnifiedDataset.addValue(scorePerIterationBack);
                AttributeInfo instanceScoreUnifiedDatasetPerIterationAtt = new AttributeInfo("instanceScoreUnifiedDatasetIteration_" + numOfIterationsBack, Column.columnType.Numeric, scorePerIterationBack, originalDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetPerIterationAtt);

                //percentile
                HashMap<Integer, TreeMap<Integer,double[]>> instancePercentileUnifiedDatasetMapPerIteration = new HashMap<>();
                instancePercentileUnifiedDatasetMapPerIteration.put(0,unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
                percentilePerIterationUnifiedDataset.addValue(calculatePercentileClassificationResults(instancePercentileUnifiedDatasetMapPerIteration, targetClassIndex, originalIndex)[0]);
            }
            else{
                AttributeInfo instanceScoreUnifiedDatasetPerIterationAttNoIterationBack = new AttributeInfo("instanceScoreUnifiedDatasetIterationNoIterationBack", Column.columnType.Numeric, -1.0, originalDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetPerIterationAttNoIterationBack);
                //how to insert -1 to the percentile calculation??
            }
        }

        //insert statistics on the previous iterations for the unified dataset
        //max
        AttributeInfo instanceScorePrevInnerIterationMaxUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMaxUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMaxUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMaxUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMaxUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMaxUnifiedDataset);
        //min
        AttributeInfo instanceScorePrevInnerIterationMinUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMinUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMinUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMinUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMinUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMinUnifiedDataset);
        //mean
        AttributeInfo instanceScorePrevInnerIterationMeanUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMeanUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMeanUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMeanUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMeanUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMeanUnifiedDataset);

        //std
        AttributeInfo instanceScorePrevInnerIterationStdUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationStdUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationStdUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationStdUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationStdUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationStdUnifiedDataset);
        //p-50
        AttributeInfo instanceScorePrevInnerIterationMedianunifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMedianunifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMedianunifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMedianunifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMedianunifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMedianunifiedDataset);

        //column data as in the dataset
        List<ColumnInfo> datasetColInfo = originalDataset.getAllColumns(false);
        DescriptiveStatistics numericAllPercentile = new DescriptiveStatistics();
        DescriptiveStatistics numericAssignedPercentile = new DescriptiveStatistics();
        DescriptiveStatistics numericHigherConfPercentile = new DescriptiveStatistics();
        DescriptiveStatistics discreteAllMode = new DescriptiveStatistics();
        DescriptiveStatistics discreteAssignedMode = new DescriptiveStatistics();
        DescriptiveStatistics discreteHigherMode = new DescriptiveStatistics();

        int numericHigherCount = 0;
        int numericRegCount = 0;
        int discreteHigherCount = 0;

        for (ColumnInfo colInf: datasetColInfo) {
            Column col = colInf.getColumn();

            if(col.getType() == Column.columnType.Numeric){
                double[] columnData = (double[])(col.getValues());
                double instancePosData = (double)(col.getValue(originalIndex));

                double maxColValue = Arrays.stream(columnData).max().getAsDouble();
                double minColValue = Arrays.stream(columnData).min().getAsDouble();
                double[] normalizedColumnData = norm100and0(maxColValue, minColValue, columnData);
                double normalizedInstancePosData = ((instancePosData - minColValue + 0.01)/(maxColValue - minColValue + 0.01))*100;

                //percentile for the instance from total column
                Percentile p = new Percentile();
                p.setData(normalizedColumnData);
                double instancePercentileColumn = p.evaluate(normalizedInstancePosData);
                numericAllPercentile.addValue(instancePercentileColumn);
//                AttributeInfo instancePercentileColumnAttr = new AttributeInfo
//                        ("instancePercentileColumn_" + instancePosRelativeIndex + "_" + currentIterationIndex, Column.columnType.Numeric, instancePercentileColumn, originalDataset.getNumOfClasses());
//                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentileColumnAttr);


                //percentile for the instance from total column - assign class and
                //percentile for the instance from total column - higher conf. level
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    TreeMap<Integer,double[]> allColumnsScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                    ArrayList<Double> assignedLabelIndeciesTemp = new ArrayList<>();
                    ArrayList<Double> higherValueIndeciesTemp = new ArrayList<>();
                    for (int i  : allColumnsScore.keySet()) {
                        if (allColumnsScore.get(i)[assignedLabel] > 0.5){
                            assignedLabelIndeciesTemp.add((double)(col.getValue(i)));
                        }
                        if(allColumnsScore.get(i)[assignedLabel] > allColumnsScore.get(originalIndex)[assignedLabel]){
                            higherValueIndeciesTemp.add((double)(col.getValue(i)));
                        }
                    }
                    double[] assignedLabelIndecies = new double[assignedLabelIndeciesTemp.size()];
                    for (int i = 0; i < assignedLabelIndeciesTemp.size(); i++) {
                        assignedLabelIndecies[i] = assignedLabelIndeciesTemp.get(i);
                    }
                    if (Arrays.stream(assignedLabelIndecies).max().isPresent() && Arrays.stream(assignedLabelIndecies).min().isPresent()){
                        double[] normAssignedLabelIndecies = norm100and0(Arrays.stream(assignedLabelIndecies).max().getAsDouble(), Arrays.stream(assignedLabelIndecies).min().getAsDouble(), assignedLabelIndecies);
                        p.setData(normAssignedLabelIndecies);
                        double assignedLabelPercentile = p.evaluate(normalizedInstancePosData);
                        numericAssignedPercentile.addValue(assignedLabelPercentile);
                        numericRegCount++;
                        double[] higherValueIndecies = new double[higherValueIndeciesTemp.size()];
                        if (higherValueIndeciesTemp.size() > 0){
                            for (int i = 0; i < higherValueIndeciesTemp.size(); i++) {
                                higherValueIndecies[i] = higherValueIndeciesTemp.get(i);
                            }
                            double[] normHigherValueIndecies = norm100and0(Arrays.stream(higherValueIndecies).max().getAsDouble(), Arrays.stream(higherValueIndecies).min().getAsDouble(), higherValueIndecies);
                            p.setData(normHigherValueIndecies);
                            double higherValuePercentile = p.evaluate(normAssignedLabelIndecies);
                            numericHigherConfPercentile.addValue(higherValuePercentile);
                            numericHigherCount++;
                        }
                    }
                }

            }
            else if (col.getType() == Column.columnType.Discrete){
                //priori
                int[] columnData = (int[])(col.getValues());
                int instancePosData = (int)(col.getValue(originalIndex));

                //mode function
                HashMap<Integer, Double> valuesCount = new HashMap<>();
                double totalValues = 0.0;
                for (int value : columnData){
                    if (valuesCount.containsKey(value)) {
                        valuesCount.put(value, valuesCount.get(value)+1.0);
                        totalValues++;
                    } else {
                        valuesCount.put(value,1.0);
                        totalValues++;
                    }
                }
                //prob to the value in the dataset
                double instanceValueCount = valuesCount.get(instancePosData);
                double instanceValueProb = instanceValueCount/totalValues;
                discreteAllMode.addValue(instanceValueProb);
//                AttributeInfo instanceValueProbAttr = new AttributeInfo
//                        ("instanceValueProb_" + originalIndex + "_" + currentIterationIndex, Column.columnType.Numeric, instanceValueProb, originalDataset.getNumOfClasses());
//                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueProbAttr);
                //Statistics on the probability of each value given all other values (in the assigned class)
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    TreeMap<Integer,double[]> allColumnsScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                    HashMap<Integer, Double> valuesCountLabeled = new HashMap<>();
                    HashMap<Integer, Double> valuesCountHigher = new HashMap<>();
                    double valueCountLabeledTotal = 0.0;
                    double valueCountHigherTotal = 0.0;
                    for (int i : allColumnsScore.keySet()) {
                        int value = (int)col.getValue(i);
                        valuesCountLabeled.put(instancePosData,1.0);
                        if (allColumnsScore.get(i)[assignedLabel] > 0.5){
                            valueCountLabeledTotal++;
                            if(valuesCountLabeled.containsKey(value)){
                                valuesCountLabeled.put(value, valuesCountLabeled.get(value)+1.0);
                            }
                            else{
                                valuesCountLabeled.put(value,1.0);
                            }
                        }
                        valuesCountHigher.put(instancePosData,1.0);
                        if(allColumnsScore.get(i)[assignedLabel] > allColumnsScore.get(originalIndex)[assignedLabel]){
                            valueCountHigherTotal++;
                            if(valuesCountHigher.containsKey(value)){
                                valuesCountHigher.put(value, valuesCountHigher.get(value)+1.0);
                            }
                            else{
                                valuesCountHigher.put(value,1.0);
                            }
                        }
                    }
                    double instanceValueLabeledCount = valuesCountLabeled.get(instancePosData);
                    double instanceValueLabeledProb = instanceValueLabeledCount/valueCountLabeledTotal;
                    double instanceValueHigherProb = -1.0;
                    try{
                        double instanceValueHigherCount = valuesCountHigher.get(instancePosData);
                        instanceValueHigherProb = instanceValueHigherCount/valueCountHigherTotal;
                    }catch (Exception e){}


                    discreteAssignedMode.addValue(instanceValueLabeledProb);
//                    AttributeInfo instanceValueLabeledProbAttr = new AttributeInfo
//                            ("instanceValueLabeledProb" + originalIndex + "_" + currentIterationIndex + '_' + partitionIndex, Column.columnType.Numeric, instanceValueLabeledProb, originalDataset.getNumOfClasses());
//                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueLabeledProbAttr);
                    if (instanceValueHigherProb > -1.0){
                        discreteHigherCount++;
                        discreteHigherMode.addValue(instanceValueHigherProb);
//                        AttributeInfo instanceValueHigherProbAttr = new AttributeInfo
//                                ("instanceValueHigherProb" + originalIndex + "_" + currentIterationIndex + '_' + partitionIndex, Column.columnType.Numeric, instanceValueHigherProb, originalDataset.getNumOfClasses());
//                        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueHigherProbAttr);
                    }

                }
            }
            //else continue;
        }
        instanceAttributesToReturn.putAll(evalDecsStats("instancePercentileColumn_" /*+ originalIndex + "_" + currentIterationIndex*/, numericAllPercentile, instanceAttributesToReturn.size()));
        double[] emptyValues ={-1.0, -1.0, -1.0, -1.0, -1.0};
        if (numericRegCount > 0){
            instanceAttributesToReturn.putAll(evalDecsStats("assignedLabelPercentile_" /*+ originalIndex + "_" + currentIterationIndex*/, numericAssignedPercentile, instanceAttributesToReturn.size()));
        }
        else{
            instanceAttributesToReturn.putAll(evalDecsStats("assignedLabelPercentile_" /*+ originalIndex + "_" + currentIterationIndex*/, new DescriptiveStatistics(emptyValues), instanceAttributesToReturn.size()));
        }

        if (numericHigherCount > 0){
            instanceAttributesToReturn.putAll(evalDecsStats("higherValuePercentile_" /*+ originalIndex + "_" + currentIterationIndex*/, numericHigherConfPercentile, instanceAttributesToReturn.size()));
        }
        else{

            instanceAttributesToReturn.putAll(evalDecsStats("higherValuePercentile_" /*+ originalIndex + "_" + currentIterationIndex*/, new DescriptiveStatistics(emptyValues), instanceAttributesToReturn.size()));
        }
        instanceAttributesToReturn.putAll(evalDecsStats("instanceValueProb_" /*+ originalIndex + "_" + currentIterationIndex*/, discreteAllMode, instanceAttributesToReturn.size()));
        instanceAttributesToReturn.putAll(evalDecsStats("instanceValueLabeledProb_" /*+ originalIndex + "_" + currentIterationIndex*/, discreteAssignedMode, instanceAttributesToReturn.size()));
        if (discreteHigherCount > 0){
            instanceAttributesToReturn.putAll(evalDecsStats("instanceValueHigherProb_" /*+ originalIndex + "_" + currentIterationIndex*/, discreteHigherMode, instanceAttributesToReturn.size()));
        }
        else{
//            double[] emptyValues ={-1.0, -1.0, -1.0, -1.0, -1.0};
            instanceAttributesToReturn.putAll(evalDecsStats("instanceValueHigherProb_" /*+ originalIndex + "_" + currentIterationIndex*/, new DescriptiveStatistics(emptyValues), instanceAttributesToReturn.size()));
        }

        //fix NaN: convert to -1.0
        for (Map.Entry<Integer,AttributeInfo> entry : instanceAttributesToReturn.entrySet()){
            AttributeInfo ai = entry.getValue();
            if (ai.getAttributeType() == Column.columnType.Numeric){
                double aiVal = (double) ai.getValue();
                if (Double.isNaN(aiVal)){
                    ai.setValue(-1.0);
                }
            }
        }
        return instanceAttributesToReturn;
    }

    private double[] norm100and0 (double max, double min, double[] arr){
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = ((arr[i] - min + 0.01)/(max - min + 0.01))*100;
        }
        return result;
    }

    private TreeMap<Integer, AttributeInfo> evalDecsStats(String attName, DescriptiveStatistics stats, int firstKey){
        TreeMap<Integer, AttributeInfo> attributesToReturn = new TreeMap<>();
        //max
        AttributeInfo attMax = new AttributeInfo
                (attName + " max", Column.columnType.Numeric, stats.getMax(), -1);
        attributesToReturn.put(firstKey, attMax);
        //min
        firstKey++;
        AttributeInfo attMin = new AttributeInfo
                (attName + " min", Column.columnType.Numeric, stats.getMin(), -1);
        attributesToReturn.put(firstKey, attMin);
        //mean
        firstKey++;
        AttributeInfo attMean = new AttributeInfo
                (attName + " mean", Column.columnType.Numeric, stats.getMean(), -1);
        attributesToReturn.put(firstKey, attMean);
        //std
        firstKey++;
        AttributeInfo attStd = new AttributeInfo
                (attName + " std", Column.columnType.Numeric, stats.getStandardDeviation(), -1);
        attributesToReturn.put(firstKey, attStd);
        //p-50
        firstKey++;
        AttributeInfo arrMedian = new AttributeInfo
                (attName + " median", Column.columnType.Numeric, stats.getPercentile(50), -1);
        attributesToReturn.put(firstKey, arrMedian);

        return attributesToReturn;
    }
}
