package com.faiss;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;

public class FaissModule extends FaissSpec {
  public static final String NAME = "Faiss";

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
    int dim=512;
    int size = 2;
    int[] ids = new int[2];
    float[][] vector = new float[size][dim];
    for(int i=0;i<size;i++){
      vector[i]= new float[512];
      ids[i]=i;
      for(int j=0;j<dim;j++){
        vector[i][j]=0.009001F*j;
      }
    }
    faissManager.faceEmbeddingIndex(vector,ids,dim);
    promise.resolve(a * b);
  }
}
