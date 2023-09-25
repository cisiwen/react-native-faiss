package com.faiss;

public class FaissManager {
  static {
    System.loadLibrary("faiss");
  }


  public void faceEmbeddingIndex(float[][] vector, int[] ids, int size){
    FaissManager.indexFromAndroid(vector,ids,size);
  }
  public static native String stringFromJNI(int a);

  public static native  String indexFromAndroid(float[][] vector, int[] ids, int size);

}
