package com.giladkz.verticalEnsemble.GeneralFunctions;

public class ClassificationItem {
    public int trueClass;
    public double[] probabilities;

    public ClassificationItem(int trueClass, double[] probabilities) {
        this.trueClass = trueClass;
        this.probabilities = probabilities;
    }

    public int getTrueClass() {
        return trueClass;
    }

    public double[] getProbabilities() {
        return probabilities;
    }

    public void setProbabilities(double[] probabilities) {
        this.probabilities = probabilities;
    }

    public void setProbabilityOfClass(int classIdx, double value) {
        this.probabilities[classIdx] = value;
    }

    public double getProbabilitiesOfClass(int index) {
        return probabilities[index];
    }
}
