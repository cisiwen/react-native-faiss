package com.faiss.models;

public class ClusteringInput {
  public boolean startOfData;
  public boolean endOfData;
  public float eps;
  public int minPts;
  public double[][] embedding;
  public int[] ids;
}
