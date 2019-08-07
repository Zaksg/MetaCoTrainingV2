package com.giladkz.verticalEnsemble.ValueFunctions;

import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;

import java.util.HashMap;
import java.util.List;

public interface ValueFunctionInterface {

    double calculateAttributeValue(Dataset dataset, List<Integer> allocatedColumnsIndices, int attributeToAnalyzeIndex,
                                   DiscretizerAbstract discretizer) throws Exception;

    HashMap<Integer, Double> Calculate_All_Candidate_Attributes_Values(Dataset dataset, List<Integer> usedAttributeIndices,
                                                                       DiscretizerAbstract discretizer, List<Integer> possibleAttributeIndices, boolean normalize) throws Exception;

    String toString();
}
