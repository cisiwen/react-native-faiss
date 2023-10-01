package com.faiss.models;

public class KNNQueryResult {
  private final int id;
  private final float score;

  public KNNQueryResult(final int id, final float score) {
    this.id = id;
    this.score = score;
  }

  public int getId() {
    return this.id;
  }

  public float getScore() {
    return this.score;
  }
}
