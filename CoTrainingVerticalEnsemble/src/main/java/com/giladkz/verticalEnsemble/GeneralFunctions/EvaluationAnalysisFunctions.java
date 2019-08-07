package com.giladkz.verticalEnsemble.GeneralFunctions;

import com.giladkz.verticalEnsemble.Data.EvaluationInfo;
import com.giladkz.verticalEnsemble.Data.EvaluationPerIteraion;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeMap;

public class EvaluationAnalysisFunctions {
    /**
     * Combines the results of several classifiers by averaging
     * @param evaluationResultsPerPartition
     * @param numOfClasses
     * @return
     */
    public static TreeMap<Integer,double[]> calculateAverageClassificationResults(HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition, int numOfClasses) {
        //double[][] resultsToReturn = new double[ evaluationResultsPerPartition.get(0).getScoreDistributions().length][numOfClasses];
        TreeMap<Integer,double[]> resultsToReturn = new TreeMap<>();
        for (int partition : evaluationResultsPerPartition.keySet()) {
            for (int i : evaluationResultsPerPartition.get(partition).getScoreDistributions().keySet()) {
                for (int j=0; j<numOfClasses; j++) {
                    if (!resultsToReturn.containsKey(i)) {
                        resultsToReturn.put(i, new double[numOfClasses]);
                    }
                    resultsToReturn.get(i)[j] += evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j];
                }
            }
        }
        //now we normalize
        for (int i : resultsToReturn.keySet()) {
            for (int j=0; j<numOfClasses; j++) {
                 resultsToReturn.get(i)[j] = resultsToReturn.get(i)[j] / numOfClasses;
            }
        }
        return normalizeClassificationResults(resultsToReturn);
    }

    /**
     * combines the results of several classifiers (partitions) by multiplication. After the multiplication
     * is complete, each confidence score is divided by the ratio of the class in the original labeled dataset
     * @param evaluationResultsPerPartition
     * @param numOfClasses
     * @param classRatios
     * @return
     */
    public static TreeMap<Integer,double[]> calculateMultiplicationClassificationResults(HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition,
                                                                   int numOfClasses, HashMap<Integer, Double> classRatios) {
        TreeMap<Integer,double[]> resultsToReturn = new TreeMap();
        for (int partition : evaluationResultsPerPartition.keySet()) {
            for (int i : evaluationResultsPerPartition.get(partition).getScoreDistributions().keySet()) {
                if (!resultsToReturn.containsKey(i)) {
                    resultsToReturn.put(i, new double[numOfClasses]);
                }

                for (int j=0; j<numOfClasses; j++) {
                    if (partition == 0) { //if its the first partition we analyze, simply assign the value
                        resultsToReturn.get(i)[j] = evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j];
                    }
                    else { //otherwise, multiply
                        resultsToReturn.get(i)[j] *= evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j];
                    }
                }
            }
        }
        for (int i: evaluationResultsPerPartition.get(0).getScoreDistributions().keySet()) {
            for (int j = 0; j < numOfClasses; j++) {
                resultsToReturn.get(i)[j] = resultsToReturn.get(i)[j] / classRatios.get(j);
            }
        }
        return normalizeClassificationResults(resultsToReturn);
    }

    /**
     * Normalizes the classifiaction results for each instance
     * @param results
     * @return
     */
    public static TreeMap<Integer,double[]> normalizeClassificationResults(TreeMap<Integer,double[]> results) {
        for (int i : results.keySet()) {
            double sum = 0;
            for (int j=0; j<results.get(i).length; j++) {
                sum += results.get(i)[j];
            }
            for (int j=0; j<results.get(i).length; j++) {
                results.get(i)[j] = results.get(i)[j]/sum;
            }
        }
        return results;
    }

    public static double[] calculatePercentileClassificationResults(HashMap<Integer, TreeMap<Integer,double[]>> iterationScoreDistPerPartition, int targetClassIndex, int instancePos){
        double[] resultsToReturn = new double[iterationScoreDistPerPartition.keySet().size()];
        for (int partition : iterationScoreDistPerPartition.keySet()) {
            int relativeIndex = -1;
            double[] scoreToPercentile = new double[iterationScoreDistPerPartition.get(partition).keySet().size()];
            int counter = 0;
            for (int instance : iterationScoreDistPerPartition.get(partition).keySet()){
                if (instance == instancePos) {
                    relativeIndex = counter; //since we convert the TreeMap to a double[], we need to know the index of the sample in the new array
                }
                double instanceScore = iterationScoreDistPerPartition.get(partition).get(instance)[targetClassIndex];
                if (instanceScore == 0.0){
                    instanceScore = 0.0001;
                }
                scoreToPercentile[counter] = instanceScore;
                counter++;
            }
            Percentile p = new Percentile();

/*            double[] normalizedScoreToPercentile = new double[scoreToPercentile.length];
            double max = Arrays.stream(normalizedScoreToPercentile).max().getAsDouble();
            double min = Arrays.stream(normalizedScoreToPercentile).min().getAsDouble();
            for (int i = 0; i < scoreToPercentile.length; i++) {
                normalizedScoreToPercentile[i] = ((scoreToPercentile[i] - min)/(max - min))*100;
            }
            double normValToCals = ((scoreToPercentile[instancePos]- min)/(max - min))*100;
            p.setData(normalizedScoreToPercentile);
            resultsToReturn[partition] = p.evaluate(normValToCals);*/

            p.setData(scoreToPercentile);
            resultsToReturn[partition] = p.evaluate(scoreToPercentile[relativeIndex]);

        }
        return resultsToReturn;
    }

}
