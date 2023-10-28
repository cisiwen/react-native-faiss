package com.faiss.clustering;
import android.os.Build;
import android.util.Log;

import com.faiss.models.DBScanInput;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;


import java.util.ArrayList;
import java.util.List;

public class DBSCANManager {

  /*
  public ArrayList<ArrayList<Integer>> ClusterFaces(DBScanInput data) {
    ArrayList<ArrayList<Integer>> idsClusters = new ArrayList<>();
    return idsClusters;
  }

  private Instances generateDoubleVectorData(DBScanInput data) {
    // Define the attributes

    int size = data.embedding.get(0).length;

    ArrayList<Attribute> attributes = new ArrayList<>();
    for(int i=1;i<=size;i++)
    {
      Attribute attribute = new Attribute("F"+i,"numeric");
      attributes.add(attribute);
    }

    int totalDataLength = data.embedding.size();
    Instances instance = new Instances("FacenetDBSCANClustering",attributes, 0);
    for(int i=0;i<totalDataLength;i++){
      double[] embedding= data.embedding.get(i);
      instance.add(new DenseInstance(size,embedding));
    }

    return instance;
  }

  public  ArrayList<ArrayList<Integer>> ClusterFaces(DBScanInput data) throws Exception {
    DBSCAN dbscan = new DBSCAN();
    dbscan.setMinPoints(data.minPts);
    dbscan.setEpsilon(data.eps);
    Instances instance= this.generateDoubleVectorData(data);
    dbscan.buildClusterer(instance);
    HashMap<Integer,ArrayList<Integer>> idsCluster = new HashMap<>();
    ArrayList<ArrayList<Integer>> idsClusters = new ArrayList<>();
    for (int i = 0; i < instance.numInstances(); i++) {
      int cluster = dbscan.clusterInstance(instance.instance(i));
      ArrayList<Integer> ids;
      if(idsCluster.containsKey(cluster)){
        ids = idsCluster.get(cluster);
      }
      else
      {
        ids = new ArrayList<>();
        idsCluster.put(cluster,ids);
      }
      ids.add(data.ids.get(i));
    }

    for (Map.Entry<Integer,ArrayList<Integer>> entry : idsCluster.entrySet()) {
      idsClusters.add(entry.getValue());
    }
    return idsClusters;
  }

*/

  public ArrayList<ArrayList<Integer>> ClusterFaces(DBScanInput data) {
    DistanceMeasure distanceMeasure = new DistanceMeasure() {
      @Override
      public double compute(double[] a, double[] b) throws DimensionMismatchException {
        EuclideanDistance distance = new EuclideanDistance();
        return distance.compute(a,b);
      }
    };

    DBSCANClusterer dbscanClusterer = new DBSCANClusterer(data.eps, data.minPts);
    List<DoublePoint> embeddings = new ArrayList<>();
    for(int i=0;i<data.embedding.size();i++) {
      embeddings.add(new DoublePoint(data.embedding.get(i)));
    }
    List<Cluster<DoublePoint>> cluster = dbscanClusterer.cluster(embeddings);
    ArrayList<ArrayList<Integer>> idsClusters = this.getClusterIds(data,cluster);
    return  idsClusters;
  }


  public int getIndexFromData(DBScanInput data, DoublePoint oneMember) {
    for (int i = 0; i < data.embedding.size(); i++) {
      DoublePoint current =  new DoublePoint(data.embedding.get(i));
      if (current.equals(oneMember)) {
        return i;
      }
    }
    return -1;
  }

  public ArrayList<ArrayList<Integer>> getClusterIds(DBScanInput data, List<Cluster<DoublePoint>> cluster) {
    ArrayList<ArrayList<Integer>> output = new ArrayList<>();
    for (int i = 0; i < cluster.size(); i++) {
      Cluster<DoublePoint> oneCluster = cluster.get(i);
      List<DoublePoint> members = oneCluster.getPoints();
      ArrayList<Integer> oneClusterIds = new ArrayList<>();
      output.add(oneClusterIds);
      Log.i("getClusterIds", String.valueOf(members.size()));
      for (int j = 0; j < members.size(); j++) {
        DoublePoint oneMember = members.get(j);
        int index = this.getIndexFromData(data, oneMember);
        oneClusterIds.add(data.ids.get(index));
      }
    }
    return output;
  }
}
