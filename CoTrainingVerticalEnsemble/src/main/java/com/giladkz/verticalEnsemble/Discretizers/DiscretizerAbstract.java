package com.giladkz.verticalEnsemble.Discretizers;

import com.giladkz.verticalEnsemble.Data.Dataset;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashMap;

public abstract class DiscretizerAbstract {

    public int numOfBins = -1;
    public HashMap<Integer, HashMap<Double, Double>> discretizationMap = new HashMap<>();

    public DiscretizerAbstract(int numOfBins) throws Exception {
        this.numOfBins = numOfBins;
    }

    /**
     * Creates the discretization map for a single attribute and saves it in a map indexed by attribute index. This
     * needs to be initialized for every attribute we wish to discretize. Only numeric columns can be discretized.
     * @param dataset
     * @param attributeToDiscretizeIndex
     * @param numOfBins
     * @throws Exception
     */
    void initializeDiscretizerForAttribute(Dataset dataset, int attributeToDiscretizeIndex, int numOfBins) throws Exception {
        throw new NotImplementedException();
    }

    /**
     * Returns the discretized value of the attribute. If the relevant attribute can't be found in the dictionary,
     * we assume it's nominal and return -1
     * @param attributeIndex
     * @param attributeValue
     * @return
     */
    int getIndex(int attributeIndex, String attributeValue) throws Exception {
        throw new NotImplementedException();
    }

    public String toString()  {
        throw new NotImplementedException();
    }
}
