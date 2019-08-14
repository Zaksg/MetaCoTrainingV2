package com.giladkz.verticalEnsemble.MetaLearning;

import com.giladkz.verticalEnsemble.Data.*;
import org.apache.commons.math3.stat.inference.TTest;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

public class ScoreDistributionBasedAttributesTdBatch extends ScoreDistributionBasedAttributes {


    public TreeMap<Integer,AttributeInfo> getScoreDistributionBasedAttributes(Dataset datasetAddedBatch
            , HashMap<Integer, EvaluationInfo> evaluationPerPartition_td, EvaluationInfo unifiedSetEvaluationResults_td
            , TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree_mainstream, EvaluationPerIteraion unifiedDatasetEvaulationResults_mainstream
            , Dataset labeledToMetaFeatures_td, Dataset unlabeledToMetaFeatures_td, int current_iteration, int targetClass, Properties properties) throws Exception{
        TreeMap<Integer,AttributeInfo> attributes = new TreeMap<>();

        //general statistics on td score dist
        TreeMap<Integer,AttributeInfo> currentScoreDistributionStatistics = new TreeMap<>();
        for (int partitionIndex : evaluationPerPartition_td.keySet()) {
            TreeMap<Integer,AttributeInfo> generalStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledToMetaFeatures_td
                    , labeledToMetaFeatures_td, evaluationPerPartition_td.get(partitionIndex).getScoreDistributions(),targetClass,"td_partition_" + partitionIndex, properties);
            for (int pos : generalStatisticsAttributes.keySet()) {
                currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), generalStatisticsAttributes.get(pos));
            }
        }
        TreeMap<Integer,AttributeInfo> unifiedStatisticsAttributes = calculateGeneralScoreDistributionStatistics(unlabeledToMetaFeatures_td
                , labeledToMetaFeatures_td, unifiedSetEvaluationResults_td.getScoreDistributions(),targetClass,"td_unified", properties);
        for (int pos : unifiedStatisticsAttributes.keySet()) {
            currentScoreDistributionStatistics.put(currentScoreDistributionStatistics.size(), unifiedStatisticsAttributes.get(pos));
        }
        attributes.putAll(currentScoreDistributionStatistics);

        //comparison per partition - T test
        TTest tTest = new TTest();
        for (int partitionIndex : evaluationPerPartition_td.keySet()) {
            double[] tdPartitionScores = new double[evaluationPerPartition_td.get(partitionIndex).getScoreDistributions().size()];
            int tdPartitionScores_cnt = 0;
            for (int instance: evaluationPerPartition_td.get(partitionIndex).getScoreDistributions().keySet()){
                tdPartitionScores[tdPartitionScores_cnt] = evaluationPerPartition_td.get(partitionIndex).getScoreDistributions().get(instance)[targetClass];
                tdPartitionScores_cnt++;
            }
            double[] msPartitionScores = new double[evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().size()];
            int msPartitionScores_cnt = 0;
            for (int instance: evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().keySet()){
                msPartitionScores[msPartitionScores_cnt] = evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex)
                        .getLatestEvaluationInfo().getScoreDistributions().get(instance)[targetClass];
                msPartitionScores_cnt++;
            }
            double TTestStatistic = tTest.t(tdPartitionScores,msPartitionScores);
            AttributeInfo tTest_att = new AttributeInfo("td_t_test_partition_"+partitionIndex, Column.columnType.Numeric, TTestStatistic, -1);
            attributes.put(attributes.size(), tTest_att);
        }
        //comparison on unified
        double[] tdUnufiedScores = new double[unifiedSetEvaluationResults_td.getScoreDistributions().size()];
        int tdUnufiedScores_cnt = 0;
        for (int instance: unifiedSetEvaluationResults_td.getScoreDistributions().keySet()){
            tdUnufiedScores[tdUnufiedScores_cnt] = unifiedSetEvaluationResults_td.getScoreDistributions().get(instance)[targetClass];
            tdUnufiedScores_cnt++;
        }
        double[] msUnufiedScores = new double[unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions().size()];
        int msUnufiedScores_cnt = 0;
        for (int instance: unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions().keySet()){
            msUnufiedScores[msUnufiedScores_cnt] = unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions().get(instance)[targetClass];
            msUnufiedScores_cnt++;
        }
        double TTestStatistic = tTest.t(tdUnufiedScores,msUnufiedScores);
        AttributeInfo tTest_att_uni = new AttributeInfo("td_t_test_unified", Column.columnType.Numeric, TTestStatistic, -1);
        attributes.put(attributes.size(), tTest_att_uni);

        //fix NaN: convert to -1.0
        for (Map.Entry<Integer,AttributeInfo> entry : attributes.entrySet()){
            AttributeInfo ai = entry.getValue();
            if (ai.getAttributeType() == Column.columnType.Numeric){
                double aiVal = (double) ai.getValue();
                if (Double.isNaN(aiVal)){
                    ai.setValue(-1.0);
                }
            }
        }
        return  attributes;
    }
}
