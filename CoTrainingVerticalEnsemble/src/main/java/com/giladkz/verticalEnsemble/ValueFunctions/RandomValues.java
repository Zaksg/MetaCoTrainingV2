package com.giladkz.verticalEnsemble.ValueFunctions;

import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;

import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.Random;

public class RandomValues implements ValueFunctionInterface {
    @Override
    public double calculateAttributeValue(Dataset dataset, List<Integer> allocatedColumnsIndices, int attributeToAnalyzeIndex, DiscretizerAbstract discretizer) throws Exception {
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        Random rnd = new Random(Integer.parseInt(properties.getProperty("randomSeed")));
        return rnd.nextDouble();
    }

    @Override
    public HashMap<Integer, Double> Calculate_All_Candidate_Attributes_Values(Dataset dataset, List<Integer> usedAttributeIndices,
                                                                              DiscretizerAbstract discretizer, List<Integer> possibleAttributeIndices, boolean normalize) throws Exception {
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        Random rnd = new Random(Integer.parseInt(properties.getProperty("randomSeed")));
        HashMap<Integer, Double> mapToReturn = new HashMap<>();
        for (int attributeIndex : possibleAttributeIndices) {
            mapToReturn.put(attributeIndex, rnd.nextDouble());
        }

        return mapToReturn;
    }

    @Override
    public String toString() {
        return "Random_Values";
    }
}
