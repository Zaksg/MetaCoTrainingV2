package com.giladkz.verticalEnsemble.FeatureSelectors;

import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;

import java.util.HashMap;
import java.util.List;

public interface FeatureSelectorInterface {

    HashMap<Integer, List<Integer>> Get_Feature_Sets(Dataset data, DiscretizerAbstract discretizer,
                              ValueFunctionInterface valueFunction, double relativeWeight, int maxNumOfSets,
                              int maxNumOfAttributesPerSet, int experimentID, int iteration, String datasetName,
                                                     boolean use_custom_feature_selection, int randomSeed) throws Exception;

    String toString();
}

