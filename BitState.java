package edu.brown.cs.burlap.tutorials.domain.simple;

import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.StateUtilities;
import burlap.mdp.core.state.UnknownKeyException;
import burlap.mdp.core.state.annotations.DeepCopyState;

import java.util.Arrays;
import java.util.List;

import static edu.brown.cs.burlap.tutorials.domain.simple.DrillModel.*;

@DeepCopyState
public class BitState implements MutableState{

    public int y1;
    public int y2;
    public int y;
    public double TIME;
    public int n;
    private final static List<Object> keys = Arrays.<Object>asList(DEPTH_IN,DEPTH_OUT,BIT_FOOTAGE,CUMM_TIME,BIT_NUMBER);

    public BitState() {
    }

    public BitState(int y1, int y2,int y, double TIME,int n) {
        this.y1 = y1;
        this.y2 = y2;
        this.y = y;
        this.TIME = TIME;
        this.n = n;
    }

    @Override
    public MutableState set(Object variableKey, Object value) {
        if(variableKey.equals(DEPTH_IN)){
            this.y1 = StateUtilities.stringOrNumber(value).intValue();
        }
        else if(variableKey.equals(DEPTH_OUT)){
            this.y2 = StateUtilities.stringOrNumber(value).intValue();
        }
        else if(variableKey.equals(BIT_FOOTAGE)){
            this.y = StateUtilities.stringOrNumber(value).intValue();
        }
        else if(variableKey.equals(CUMM_TIME)) {
            this.TIME = StateUtilities.stringOrNumber(value).doubleValue();
        }
        else if(variableKey.equals(BIT_NUMBER)) {
            this.n = StateUtilities.stringOrNumber(value).intValue();
        }
        else{
            throw new UnknownKeyException(variableKey);
        }
        return this;
    }

    public List<Object> variableKeys() {
        return keys;
    }

    @Override
    public Object get(Object variableKey) {
        if(variableKey.equals(DEPTH_IN)){
            return y1;
        }
        else if(variableKey.equals(DEPTH_OUT)){
            return y2;
        }
        else if(variableKey.equals(BIT_FOOTAGE)){
            return y;
        }
        else if(variableKey.equals(CUMM_TIME)) {
            return TIME;
        }
        else if(variableKey.equals(BIT_NUMBER)) {
            return n;
        }
        else
        throw new UnknownKeyException(variableKey);
    }

    @Override
    public BitState copy() {
        return new BitState(y1,y2,y,TIME,n);
    }

    @Override
    public String toString() {
        return StateUtilities.stateToString(this);
    }
}
