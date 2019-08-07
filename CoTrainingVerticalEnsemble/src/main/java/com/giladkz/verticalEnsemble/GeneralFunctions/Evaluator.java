package com.giladkz.verticalEnsemble.GeneralFunctions;

import com.giladkz.verticalEnsemble.Data.ColumnInfo;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Operators.OperatorAssignment;

import java.util.List;
import java.util.Properties;

public interface Evaluator {
    enum evaluatorType { Filter, Wrapper }

    /**
     * Enables us to know the evaluation criteria used. If the filter and wrapper use the same criteria
     * it is possible to conduct more advanced comparisons of expected vs actual performance
     */
    enum evaluatorScoringMethod { AUC, InformationGain, ProbDiff, LogLoss, ClassifierProbability}

    double produceScore(Dataset analyzedDatasets, ClassificationResults currentScore, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception;

    double produceAverageScore(List<Dataset> analyzedDatasets, List<ClassificationResults> classificationResults, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception;

    evaluatorType getType();

    evaluatorScoringMethod getEvaluatorScoreingMethod();
}
