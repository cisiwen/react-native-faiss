package com.faiss;

import com.faiss.models.IndexInput;

public class FaissManager {
  static {
    System.loadLibrary("faiss");
  }


  public String faceEmbeddingIndex(IndexInput input){
    return FaissManager.indexFromAndroid(input.embedding,input.ids,input.dim);
  }
  public static native String stringFromJNI(int a);

  public static native  String indexFromAndroid(float[][] vector, int[] ids, int size);

}
