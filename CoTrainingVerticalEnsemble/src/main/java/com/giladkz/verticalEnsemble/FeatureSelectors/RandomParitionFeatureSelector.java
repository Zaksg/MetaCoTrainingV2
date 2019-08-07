package com.giladkz.verticalEnsemble.FeatureSelectors;

import com.giladkz.verticalEnsemble.Data.ColumnInfo;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;

import java.io.InputStream;
import java.util.*;

public class RandomParitionFeatureSelector implements FeatureSelectorInterface {
    @Override
    public HashMap<Integer, List<Integer>> Get_Feature_Sets(Dataset data, DiscretizerAbstract discretizer,
                        ValueFunctionInterface valueFunction, double relativeWeight, int maxNumOfSets,
                        int maxNumOfAttributesPerSet, int experimentID, int iteration, String datasetName,
                        boolean use_custom_feature_selection,  int randomSeed) throws Exception {

        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        Random rnd = new Random(randomSeed);
        HashMap<Integer, List<Integer>> mapToReturn = new HashMap<>();
        List<ColumnInfo> columnsToPartition = data.getAllColumns(false);
        List<ColumnInfo> allocatedColumns = new ArrayList<>();

        int currentSetIndex = 0;
        while (allocatedColumns.size() < columnsToPartition.size()) {
            boolean found = false;
            while (!found) {
                int pos = rnd.nextInt(columnsToPartition.size());
                ColumnInfo ci = columnsToPartition.get(pos);
                if (!allocatedColumns.contains(ci)) {
                    if (!mapToReturn.containsKey(currentSetIndex)) {
                        mapToReturn.put(currentSetIndex, new ArrayList<>());
                    }
                    mapToReturn.get(currentSetIndex).add(pos);
                    allocatedColumns.add(ci);
                    found = true;
                }
            }
            currentSetIndex++;
            currentSetIndex = (currentSetIndex % maxNumOfSets);
        }

        return mapToReturn;
    }

    @Override
    public String toString() {
        return "RandomParitionFeatureSelector";
    }
}
