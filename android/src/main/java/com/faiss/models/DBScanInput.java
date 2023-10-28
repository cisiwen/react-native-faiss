package com.faiss.models;

/*
import org.apache.commons.math3.ml.clustering.DoublePoint;
*/
import java.util.ArrayList;

public class DBScanInput {

  public float eps;
  public int minPts;

  public ArrayList<double[]> embedding;
  public ArrayList<Integer> ids;
  public DBScanInput(){
    this.embedding = new ArrayList<double[]>();
    this.ids = new ArrayList<Integer>();
  }
}
