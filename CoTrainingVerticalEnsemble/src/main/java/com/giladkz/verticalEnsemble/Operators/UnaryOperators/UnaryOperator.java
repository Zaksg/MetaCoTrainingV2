package com.giladkz.verticalEnsemble.Operators.UnaryOperators;

import com.giladkz.verticalEnsemble.Data.ColumnInfo;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Operators.Operator;

import java.util.List;

/**
 * Created by giladkatz on 20/02/2016.
 */
public abstract class UnaryOperator extends Operator {

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        //if there are any target columns or if there is more than one source column, return false
        if (sourceColumns.size() != 1 || (targetColumns != null && targetColumns.size() != 0)) {
            return false;
        }
        else {
            return true;
        }
    }

    public Operator.operatorType getType() {
        return Operator.operatorType.Unary;
    }

    public abstract Operator.outputType requiredInputType();

    public abstract int getNumOfBins();
}
