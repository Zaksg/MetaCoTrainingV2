package com.giladkz.verticalEnsemble.Data;

import java.util.*;

public class FoldsInfo {

    private int _numOfTrainingFolds;
    private int _numOfValidationFolds;
    private int _numOfTestFolds;
    private double _trainingSetPercentage;
    private int _numOfTrainingInstances;
    private double _validationSetPercentage;
    private int _numOfValidationInstances;
    private double _testSetPercentage;
    private int _numOfTestSetInstances;
    private boolean _maintainClassRatio;
    public enum foldType { Train, Validation, Test }
    public foldType _assignRemainingInstancesToFold;



    /**
     *
     * @param numOfTrainingFolds
     * @param numOfValidationFolds can only be 0 or 1
     * @param numOfTestFolds can only be 0 or 1
     * @param trainingSetPercentage
     * @param numOfTraningInstances
     * @param validationSetPercentage
     * @param numOfValidationSetInstances
     * @param maintainClassRatio
     * @param assignRemainingInstancesToFold The fold to which the "leftover" instances (those that are left after the partitioning to the various folds is done) are assigned to
     */
    public FoldsInfo(int numOfTrainingFolds, int numOfValidationFolds, int numOfTestFolds, double trainingSetPercentage, int numOfTraningInstances, double validationSetPercentage,
                     int numOfValidationSetInstances, double testSetPercentage, int numOfTestSetInstances, boolean maintainClassRatio, foldType assignRemainingInstancesToFold) throws Exception {
        _numOfTrainingFolds = numOfTrainingFolds;
        _numOfValidationFolds = numOfValidationFolds;
        _numOfTestFolds = numOfTestFolds;
        _trainingSetPercentage = trainingSetPercentage;
        _numOfTrainingInstances = numOfTraningInstances;
        _validationSetPercentage = validationSetPercentage;
        _numOfValidationInstances = numOfValidationSetInstances;
        _maintainClassRatio = maintainClassRatio;
        _testSetPercentage = testSetPercentage;
        _numOfTestSetInstances = numOfTestSetInstances;
        _assignRemainingInstancesToFold = assignRemainingInstancesToFold;

        if ((_numOfValidationFolds > 1 || _numOfValidationFolds < 0) || (_numOfTestFolds > 1 || _numOfTestFolds <0)) {
            throw new Exception("There can only be 0 or 1 validation and test folds");
        }
    }

    public int getNumOfTrainingFolds() {
        return _numOfTrainingFolds;
    }

    public int getNumOfValidationFolds() {
        return _numOfValidationFolds;
    }

    public int getNumOfTestFolds() {
        return _numOfTestFolds;
    }

    public double getTraninigSetPercentage() {
        return _trainingSetPercentage;
    }

    public int getNumOfTrainingInstances() {
        return _numOfTrainingInstances;
    }

    public double getValidationSetPercentage() {
        return _validationSetPercentage;
    }

    public int getNumOfValidationInstances() {
        return _numOfValidationInstances;
    }

    public double getTestSetPercentage() {
        return _testSetPercentage;
    }

    public int getNumOfTestInstances() {
        return _numOfTestSetInstances;
    }

    public boolean getMaintainClassRatio() {
        return _maintainClassRatio;
    }

    public foldType getAssignRemainingInstancesToFold() {
        return _assignRemainingInstancesToFold;
    }

    /**
     * Returns the number of folds of each type
     * @return
     */
    public HashMap<foldType, Integer> getNumOfFolds() {
        HashMap<foldType, Integer> mapToReturn = new HashMap<>();
        mapToReturn.put(foldType.Train, _numOfTrainingFolds);
        mapToReturn.put(foldType.Validation, _numOfValidationFolds);
        mapToReturn.put(foldType.Test, _numOfTestFolds);
        return mapToReturn;
    }

    /**
     * Returns a map with the actual number of instance PER FOLD for each type of fold
     * @param numOfDatasetInstances
     * @return
     * @throws Exception
     */
    public HashMap<foldType,Integer> getNumOfInstacesPerFold(int numOfDatasetInstances) throws Exception {
        int numOfTrainingFoldInstances = Math.max( Math.max(0, _numOfTrainingInstances), Math.max(0, (int)(_trainingSetPercentage*numOfDatasetInstances/_numOfTrainingFolds)));

        int numOfValidationFoldInstances = 0;
        if (_numOfValidationFolds > 0) {
            numOfValidationFoldInstances = Math.max( Math.max(0, _numOfValidationInstances), Math.max(0, (int)_validationSetPercentage*numOfDatasetInstances));
        }

        int numOfTestFoldInstances = 0;
        if (_numOfTestFolds > 0) {
            numOfTestFoldInstances = Math.max( Math.max(0, _numOfTestSetInstances), Math.max(0, (int)(_testSetPercentage*numOfDatasetInstances)));
        }

        int totalNumOfAllocatedInstences = (numOfTrainingFoldInstances * _numOfTrainingFolds) + numOfValidationFoldInstances + numOfTestFoldInstances;

        // If we over or under 1% of the instances then throw an exception
        if (totalNumOfAllocatedInstences < 0.99*numOfDatasetInstances || totalNumOfAllocatedInstences > 1.01*numOfDatasetInstances) {
            throw new Exception("instance allocation is incorrect. Please re-check");
        }

        HashMap<foldType,Integer> instancesAllocationMap = new HashMap<>();
        instancesAllocationMap.put(foldType.Train, numOfTrainingFoldInstances);
        instancesAllocationMap.put(foldType.Validation, numOfValidationFoldInstances);
        instancesAllocationMap.put(foldType.Test, numOfTestFoldInstances);
        return instancesAllocationMap;
    }

    /**
     * Used to determine the number of instances PER FOLD for each type of fold while also returning the NUMBER OF INSTANCES PER CLASS
     * (if the class ratio does not need to be maintained, the getNumOfInstacesPerFold can be called directly).
     * @param itemIndicesByClass
     * @param numOfDatasetInstances
     * @return
     */
    public HashMap<foldType, HashMap<Integer,Integer>> getNumOfInstancesPerFoldPerClass(ArrayList<ArrayList<Integer>> itemIndicesByClass,
                                                                                        int numOfDatasetInstances) throws Exception {
        //start by getting the total number of instances per fold
        HashMap<foldType,Integer> totalNumOfInstancesPerFold = getNumOfInstacesPerFold(numOfDatasetInstances);

        //perform the conversion to percentages
        HashMap<Integer,Double> classPercentages = new HashMap<>();
        for (int i=0; i<itemIndicesByClass.size(); i++) {
            classPercentages.put(i, ((double)itemIndicesByClass.get(i).size())/numOfDatasetInstances);
        }

        HashMap<foldType, HashMap<Integer,Integer>> mapToReturn = new HashMap<>();

        for (foldType fold : totalNumOfInstancesPerFold.keySet()) {
            mapToReturn.put(fold, new HashMap<>());
            for (int i=0; i<itemIndicesByClass.size(); i++) {
                mapToReturn.get(fold).put(i, (int)Math.round(totalNumOfInstancesPerFold.get(fold)*classPercentages.get(i)));
            }
        }

        return mapToReturn;
    }
}
