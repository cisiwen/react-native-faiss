package com.faiss;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.faiss.clustering.DBSCANManager;
import com.faiss.clustering.DBScanELKIManager;
import com.faiss.models.ClusteringInput;
import com.faiss.models.ClusteringOutput;
import com.faiss.models.DBScanInput;
import com.faiss.models.IndexInput;
import com.faiss.models.KNNQueryResult;
import com.faiss.models.KmeanOutputItem;
import com.faiss.models.QueryInput;
import com.faiss.models.TrainIndexInput;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/*
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
*/
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class FaissModule extends FaissSpec {
  public static final String NAME = "Faiss";

  public DBScanInput dbScanInput;

  FaissModule(ReactApplicationContext context) {
    super(context);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }


  // Example method
  // See https://reactnative.dev/docs/native-modules-android
  @ReactMethod
  public void multiply(double a, double b, Promise promise) {
    FaissManager faissManager = new FaissManager();
    int dim = 512;
    int size = 2;
    int[] ids = new int[2];
    float[][] vector = new float[size][dim];
    for (int i = 0; i < size; i++) {
      vector[i] = new float[512];
      ids[i] = i;
      for (int j = 0; j < dim; j++) {
        vector[i][j] = 0.009001F * j;
      }
    }
    IndexInput input = new IndexInput();
    input.embedding = vector;
    input.dim = dim;
    input.ids = ids;
    //faissManager.faceEmbeddingIndex(input);
    promise.resolve(a * b);
  }

  @ReactMethod
  public void faissIndex(String data, Promise promise) {
    Type inputType = new TypeToken<IndexInput>() {
    }.getType();
    Gson gson = new Gson();
    IndexInput input = gson.fromJson(data, inputType);
    FaissManager faissManager = new FaissManager();
    String result = faissManager.faceEmbeddingIndex(input);
    promise.resolve(result);
  }


  @ReactMethod
  public void queryIndex(String query, Promise promise) {
    Type queryInputType = new TypeToken<QueryInput>() {
    }.getType();
    Gson gson = new Gson();
    QueryInput input = gson.fromJson(query, queryInputType);
    FaissManager faissManager = new FaissManager();
    KNNQueryResult[] results = faissManager.queryIndex(input);
    promise.resolve(gson.toJson(results));
  }


  @ReactMethod
  public void dbscanClustering(String data, Promise promise) {
    Type inputType = new TypeToken<ClusteringInput>() {
    }.getType();
    Gson gson = new Gson();
    ClusteringInput input = gson.fromJson(data, inputType);
    if (input.startOfData) {
      this.dbScanInput = new DBScanInput();
    }
    for (int i = 0; i < input.embedding.length; i++) {
      this.dbScanInput.embedding.add(input.embedding[i]);
      this.dbScanInput.ids.add(input.ids[i]);
    }
    if (input.endOfData) {
      //DBScanELKIManager dbscanManager = new DBScanELKIManager();
      DBSCANManager dbscanManager = new DBSCANManager();
      this.dbScanInput.eps = input.eps;
      this.dbScanInput.minPts = input.minPts;
      ClusteringOutput output = new ClusteringOutput();
      try {
        ArrayList<ArrayList<Integer>> clusters = dbscanManager.ClusterFaces(this.dbScanInput);
        output.clusters = clusters;
        output.totalData = this.dbScanInput.embedding.size();
        promise.resolve(gson.toJson(output));
      } catch (Exception ex) {
        promise.reject(ex);
      }


    } else {
      ClusteringOutput output = new ClusteringOutput();
      output.totalData = this.dbScanInput.embedding.size();
      promise.resolve(gson.toJson(output));
    }
  }

  @ReactMethod
  public  void clusterWithFile(String fileUri, float eps, int minPts,Promise promise) {
    Gson gson = new Gson();
    try {
      DBSCANManager dbscanManager = new DBSCANManager();
      ArrayList<ArrayList<Integer>> clusters = dbscanManager.ClusterWithFile(fileUri, eps, minPts,this.getReactApplicationContext());
      ClusteringOutput output = new ClusteringOutput();
      output.clusters = clusters;
      output.totalData = this.dbScanInput.embedding.size();
      promise.resolve(gson.toJson(output));
    } catch (Exception ex) {
      promise.reject(ex);
    }
  }

  @ReactMethod
  public void kmeansCluster(String fileUri,int k, int dim,int size,Promise promise){
    try {
      FaissManager faissManager = new FaissManager();
      KmeanOutputItem[] result = faissManager.faissKmeans(fileUri, k, dim, size);
      Gson gson = new Gson();
      promise.resolve(gson.toJson(result));
    }
    catch (Exception ex){
      promise.reject(ex);
    }

  }

  @ReactMethod
  public void trainIndex(String trainInput, Promise promise) {
    Type trainInputType = new TypeToken<TrainIndexInput>() {
    }.getType();
    Gson gson = new Gson();
    TrainIndexInput input = gson.fromJson(trainInput, trainInputType);
    FaissManager faissManager = new FaissManager();
    byte[] result = faissManager.trainIndex(input);
    promise.resolve(gson.toJson(result));
  }
}
