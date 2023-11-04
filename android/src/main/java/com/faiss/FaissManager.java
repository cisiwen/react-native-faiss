package com.faiss;

import com.faiss.models.IndexInput;
import com.faiss.models.KNNQueryResult;
import com.faiss.models.QueryInput;
import com.faiss.models.TrainIndexInput;

public class FaissManager {
  static {
    System.loadLibrary("faiss");
  }


  public String faceEmbeddingIndex(IndexInput input){
    if(input.template!=null){
       return FaissManager.CreateIndexFromTemplate(input.ids,input.embedding, input.indexFullName,input.template);
    }
    else {
      return FaissManager.indexFromAndroid(input.embedding, input.ids, input.dim, input.indexFullName,input.ended);
    }
  }

  public String buildFaissIndex(String filePath){
      return  null;
  }


  public KNNQueryResult[] queryIndex(QueryInput input) {
    return  FaissManager.QueryIndex(input.indexFullName,input.queryVector,input.k);
  }




  public byte[] trainIndex(TrainIndexInput input){
    long trainPointer1 = TransferVectors(0, input.embedding,input.dim);
    byte[] result = FaissManager.TrainIndex(input.dim,trainPointer1);
    FreeVectors(trainPointer1);
    return  result;
  }

  public  int[] faissKmeans(String dataFile,int k, int dim, int size) {
    return FaissManager.KmeansCluster(dataFile,k,dim,size);
  }

  public static native String stringFromJNI(int a);

  public static native  String indexFromAndroid(float[][] vector, int[] ids, int size,String indexName,String ended);

  public  static native  KNNQueryResult[] QueryIndex(String indexName,float[] queryVector,int k);

  public  static native  byte[] TrainIndex(int dimj,long trainVectorsPointer);
  public static native long TransferVectors(long vectorsPointer, float[][] trainingData,int dim);

  public static native void FreeVectors(long vectorsPointer);

  public  static  native  String CreateIndexFromTemplate(int[] idsJ, float[][] vectorsJ, String indexPathJ, byte[] templateIndexJ);

  public  static  native  int[] KmeansCluster(String indexName,int k,int dim,int size);
}
