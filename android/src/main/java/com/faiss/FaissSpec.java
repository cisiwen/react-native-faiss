package com.faiss;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.Promise;

abstract class FaissSpec extends ReactContextBaseJavaModule {
  FaissSpec(ReactApplicationContext context) {
    super(context);
  }

  public abstract void multiply(double a, double b, Promise promise);
  public abstract void faissIndex(String data,Promise promise);

  public abstract void queryIndex(String data,Promise promise);

  public abstract void trainIndex(String trainInput,Promise promise);
  public abstract void  dbscanClustering(String data, Promise promise) throws Exception;
}
