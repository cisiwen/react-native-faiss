package com.faiss;

import com.faiss.models.IndexInput;
import com.faiss.models.KNNQueryResult;
import com.faiss.models.QueryInput;

public class FaissManager {
  static {
    System.loadLibrary("faiss");
  }


  public String faceEmbeddingIndex(IndexInput input){
    return FaissManager.indexFromAndroid(input.embedding,input.ids,input.dim,input.indexFullName);
  }


  public KNNQueryResult[] queryIndex(QueryInput input) {
    return  FaissManager.QueryIndex(input.indexFullName,input.queryVector,input.k);
  }
  public static native String stringFromJNI(int a);

  public static native  String indexFromAndroid(float[][] vector, int[] ids, int size,String indexName);

  public  static native  KNNQueryResult[] QueryIndex(String indexName,float[] queryVector,int k);

}
