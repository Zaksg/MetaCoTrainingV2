package com.giladkz.verticalEnsemble.Data;

import weka.classifiers.evaluation.Evaluation;

import java.io.InputStream;
import java.util.*;

public class EvaluationInfo {
    private Properties properties;

    private Evaluation evaluation;
    private TreeMap<Integer,double[]> scoreDistPerInstance;
    private ArrayList<Integer> instanceIndices;

    //This object contains the sorted confidence scores. It will be generated the first time and then used in consecutive calls
    HashMap<Integer, TreeMap<Double,List<Integer>>> sortedConfidenceScoresByClass = new HashMap<>(); //class index -> Confidence score -> List of instance indices


    public EvaluationInfo(Evaluation evaluationStats, double[][] scoreDistributions, ArrayList<Integer> instanceIndices) throws Exception {
        this.evaluation = evaluationStats;
        this.instanceIndices = instanceIndices;

        //Our reason for this implementation is as follows: there are going to be different numbers of samples for different iterations. The
        //only way to know which instance appears in every iteration is to save the indices
        this.scoreDistPerInstance = new TreeMap<>();
        Collections.sort(instanceIndices);
        for (int i=0; i<scoreDistributions.length; i++) {
            int instanceIndex = instanceIndices.get(i);
            scoreDistPerInstance.put(instanceIndex,scoreDistributions[i]);
        }

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
    }

    public Evaluation getEvaluationStats() {return evaluation;}

    public TreeMap<Integer,double[]> getScoreDistributions() {return scoreDistPerInstance;}

    /**
     * Returns the instances sorted by their confidence scores (for the class provided). The Instances are sorted in
     * a DESCENDING order.
     * to the requested class
     * @param classIndex
     * @return
     */
    public TreeMap<Double,List<Integer>> getTopConfidenceInstancesPerClass(int classIndex) {
        if (!sortedConfidenceScoresByClass.containsKey(classIndex)) {
            generateSortedConfidenceScoresMap(classIndex);
        }

        return sortedConfidenceScoresByClass.get(classIndex);
    }

    /**
     * Populates the sortedConfidenceScoresByClass object with the confidence scores of all instances for
     * a given class
     * @param classIndex
     */
    public void generateSortedConfidenceScoresMap(int classIndex) {
        for (int key : scoreDistPerInstance.keySet()) {
            double confidenceScore = scoreDistPerInstance.get(key)[classIndex];
            if (!sortedConfidenceScoresByClass.containsKey(classIndex)) {
                sortedConfidenceScoresByClass.put(classIndex, new TreeMap<>(Collections.reverseOrder()));
            }
            if (!sortedConfidenceScoresByClass.get(classIndex).containsKey(confidenceScore)) {
                sortedConfidenceScoresByClass.get(classIndex).put(confidenceScore, new ArrayList<>());
            }
            sortedConfidenceScoresByClass.get(classIndex).get(confidenceScore).add(key);
        }
    }


}