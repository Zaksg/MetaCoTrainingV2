package com.giladkz.verticalEnsemble.Discretizers;

import com.giladkz.verticalEnsemble.Data.Column;
import com.giladkz.verticalEnsemble.Data.Dataset;

import java.util.Arrays;
import java.util.HashMap;

public class EqualRangeBinsDiscretizer extends DiscretizerAbstract {

    public EqualRangeBinsDiscretizer(int numOfBins) throws Exception {
        super(numOfBins);
    }

    @Override
    public void initializeDiscretizerForAttribute(Dataset dataset, int attributeToDiscretizeIndex, int numOfBins) throws Exception {
        this.numOfBins = numOfBins;

        double min_value = Double.MAX_VALUE;
        double max_value = Double.MIN_VALUE;

        if (dataset.getColumns(Arrays.asList(attributeToDiscretizeIndex)).get(0).getColumn().getType() == Column.columnType.Numeric) {
            double[] values = (double[])dataset.getColumns(Arrays.asList(attributeToDiscretizeIndex)).get(0).getColumn().getValues();
            for (double value : values) {
                if (value < min_value)
                    min_value = value;
                if (value > max_value)
                    max_value = value;
            }

            HashMap<Double,Double> tempMap = new HashMap<>();
            double binWidth = (max_value - min_value)/numOfBins;
            double current_value = min_value;
            for (int i = 0; i < numOfBins; i++)
            {
                if (tempMap.containsKey(current_value))
                {
                    //this can only happen when the min and the max are equal - in this case there is nothing to do - just return
                    break;
                }
                tempMap.put(current_value, current_value + binWidth);
                current_value += binWidth;
            }
            discretizationMap.put(attributeToDiscretizeIndex, tempMap);
        }
        else {
            throw new Exception("Cannot discretize a non-numeric column");
        }
    }



    @Override
    public int getIndex(int attributeIndex, String attributeValue) throws Exception {
        if (discretizationMap == null || !discretizationMap.containsKey(attributeIndex)) {
            throw new Exception("The discretizer has not been initialized");
        }

        int index = 0;

        double attVal;

        try {
            attVal = Double.parseDouble(attributeValue);
        }
        catch (NumberFormatException exception) {
            if (attributeValue.equals("?"))
                return -1;
            else
                throw new Exception("problem parsing a numeric value");
        }

        for (double lowerBound : discretizationMap.get(attributeIndex).keySet()) {
            if (lowerBound >= attVal)
            {
                return index;
            }
            index++;
        }
        return index;
    }

    @Override
    public String toString()
    {
        return "Equal_Range_Bins_Discretizer_" + numOfBins + "_bins";
    }
}
