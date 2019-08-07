package com.giladkz.verticalEnsemble.Data;

import com.giladkz.verticalEnsemble.Data.Column;

/**
 * Created by giladkatz on 07/05/2016.
 */
public class AttributeInfo {
    private String attributeName;
    private Column.columnType attributeType;
    private Object value;
    private int numOfDiscreteValues;

    public AttributeInfo(String attName, Column.columnType attType, Object attValue, int numOfValues) {
        this.attributeName = attName;
        this.attributeType = attType;
        this.value = attValue;
        this.numOfDiscreteValues = numOfValues;
    }

    public String getAttributeName() {
        return attributeName;
    }

    public Column.columnType getAttributeType() {
        return attributeType;
    }

    public Object getValue() {
        return value;
    }

    public int getNumOfDiscreteValues() {
        return numOfDiscreteValues;
    }

    public void setValue(Object value) {
        this.value = value;
    }
}