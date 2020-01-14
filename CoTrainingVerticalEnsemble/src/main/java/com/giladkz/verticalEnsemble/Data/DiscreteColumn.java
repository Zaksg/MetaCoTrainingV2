package com.giladkz.verticalEnsemble.Data;

import java.io.Serializable;

/**
 * Created by giladkatz on 11/02/2016.
 */
public class DiscreteColumn implements Column,Serializable {
    private int[] values;
    private int numOfPossibleValues;

    public DiscreteColumn(int size, int numOfPossibleValues) {
        values = new int[size];
        this.numOfPossibleValues = numOfPossibleValues;
    }

    public Object getValue(int i) { return values[i]; }

    public void setValue(int i, Object obj) {
        values[i] = (Integer)obj;
    }

    public columnType getType() {return columnType.Discrete;}

    public int getNumOfInstances() {
        return values.length;
    }

    public int getNumOfPossibleValues() {
        return this.numOfPossibleValues;
    }

    public void setAllValues(Object v) {
        values = (int[])v;
    }

    public Object getValues() {return values;}

}
