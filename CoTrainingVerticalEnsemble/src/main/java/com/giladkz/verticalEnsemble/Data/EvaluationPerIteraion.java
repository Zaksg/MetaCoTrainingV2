package com.giladkz.verticalEnsemble.Data;

import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

public class EvaluationPerIteraion {

    TreeMap<Integer, EvaluationInfo> evaluationPerIterationMap = new TreeMap<>();


    public EvaluationPerIteraion() {

    }

    public void addEvaluationInfo(EvaluationInfo ei, int iterationIndex) {
        evaluationPerIterationMap.put(iterationIndex, ei);
    }

    public EvaluationInfo getLatestEvaluationInfo() {
        return evaluationPerIterationMap.get(evaluationPerIterationMap.lastKey());
    }

    /**
     * Returns the latest X iterations
     * @param numOfEvaluations
     * @return
     */
    public TreeMap<Integer,EvaluationInfo> getLastXEvaluations(int numOfEvaluations) {
        return null;
    }

    public EvaluationInfo getIterationEvaluationInfo(int iterationIndex) {
        return evaluationPerIterationMap.get(iterationIndex);
    }


    public TreeMap<Integer, EvaluationInfo> getEvaluationInfoByIndices(List<Integer> indices) {
        TreeMap<Integer, EvaluationInfo> mapToReturn = new TreeMap<>();
        for (int index: indices) {
            mapToReturn.put(index, evaluationPerIterationMap.get(index));
        }
        return mapToReturn;
    }

    public TreeMap<Integer, EvaluationInfo> getEvaluationInfoByStartAndFinishIndices(int startIndex, int endIndex) {
        TreeMap<Integer, EvaluationInfo> mapToReturn = new TreeMap<>();
        for (int i=startIndex; i<=endIndex; i++) {
            mapToReturn.put(i, evaluationPerIterationMap.get(i));
        }

        return mapToReturn;
    }

    public TreeMap<Integer, EvaluationInfo> getAllIterations() {
        return evaluationPerIterationMap;
    }

    public TreeMap<Integer, TreeMap<Integer,double[]>> getAllIterationsScoreDistributions() {
        //return evaluationPerIterationMap;
        TreeMap<Integer, TreeMap<Integer,double[]>> mapToReturn = new TreeMap<>();
        for (int i : evaluationPerIterationMap.keySet()) {
            mapToReturn.put(i, evaluationPerIterationMap.get(i).getScoreDistributions());
        }
        return mapToReturn;
    }
}
