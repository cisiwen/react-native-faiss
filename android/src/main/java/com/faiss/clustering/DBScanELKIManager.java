package com.faiss.clustering;

import android.util.Log;

import com.faiss.models.DBScanInput;
import com.google.gson.Gson;

import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import elki.clustering.dbscan.DBSCAN;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.StaticArrayDatabase;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDRange;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDs;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.datasource.ArrayAdapterDatabaseConnection;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;

public class DBScanELKIManager {

  public int getIndexFromData(DBScanInput data, NumberVector oneMember) {

    for (int i = 0; i < data.embedding.size(); i++) {
      double[] current= data.embedding.get(i);
      DoublePoint cvector = new DoublePoint(current);
      DoublePoint ovector = new DoublePoint(oneMember.toArray());
      if(cvector.equals(ovector)){
        return  i;
      }
    }
    return -1;
  }
  public  ArrayList<ArrayList<Integer>> ClusterFaces(DBScanInput data){
    double[][] embedding =new double[data.embedding.size()][data.embedding.get(0).length];
    String[] labels = new String[data.ids.size()];
    for(int i=0;i<data.embedding.size();i++){
      embedding[i]=data.embedding.get(i);
      labels[i]=data.ids.get(i).toString();
    }
    ArrayAdapterDatabaseConnection dbc = new ArrayAdapterDatabaseConnection(embedding);
    StaticArrayDatabase database = new StaticArrayDatabase(dbc, null);
    database.initialize();
    Relation<NumberVector> rel = database.getRelation(TypeUtil.NUMBER_VECTOR_FIELD);
    DBSCAN<DoubleVector> dbscan = new DBSCAN<>(new EuclideanDistance(), data.eps, data.minPts);
    Clustering<Model> result = dbscan.autorun(database);
    List<Cluster<Model>> clusters = result.getAllClusters();
    DBIDRange ids = (DBIDRange) rel.getDBIDs();
    ArrayList<ArrayList<Integer>> output = new ArrayList<>();
    Log.i("INPUTIDS",data.ids.toString());
    Log.i("DBIDS",rel.getDBIDs().toString());
    for(Cluster<Model> cluster : clusters) {
      ArrayList<Integer> clusterIds = new ArrayList<>();
      output.add(clusterIds);
      Log.i("clusterstart","clusterstart");
      for(DBIDIter it = cluster.getIDs().iter(); it.valid(); it.advance()) {
        // To get the vector use:
        //NumberVector v = rel.get(it);
        //int id = this.getIndexFromData(data,v);
        final int offset = ids.getOffset(it);
        Log.i("getIndexFromData",String.valueOf(offset));
        clusterIds.add(offset);
      }
      Log.i("clusterstart","clusterend");
    }
    dbscan=null;
    database= null;
    dbc=null;
    result=null;
    clusters=null;
    return output;
  }
}
