package com.faiss;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.faiss.models.IndexInput;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

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
    IndexInput input = new IndexInput();
    input.embedding= vector;
    input.dim= dim;
    input.ids= ids;
    //faissManager.faceEmbeddingIndex(input);
    promise.resolve(a * b);
  }

  @ReactMethod
  public void faissIndex(String data, Promise promise) {
    Type inputType = new TypeToken<IndexInput>(){}.getType();
    Gson gson = new Gson();
    IndexInput input = gson.fromJson(data,inputType);
    FaissManager faissManager = new FaissManager();
    String result = faissManager.faceEmbeddingIndex(input);
    promise.resolve(result);
  }
}
