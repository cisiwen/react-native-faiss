package com.faiss.clustering;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.util.Log;

import com.faiss.models.DBScanInput;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DBSCANManager {

  class IORejectionException extends Exception {
    private String code;

    public IORejectionException(String code, String message) {
      super(message);
      this.code = code;
    }

    public String getCode() {
      return code;
    }
  }
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

  private Uri getFileUri(String filepath, boolean isDirectoryAllowed) throws IORejectionException {
    Uri uri = Uri.parse(filepath);
    if (uri.getScheme() == null) {
      // No prefix, assuming that provided path is absolute path to file
      File file = new File(filepath);
      if (!isDirectoryAllowed && file.isDirectory()) {
        throw new IORejectionException("EISDIR", "EISDIR: illegal operation on a directory, read '" + filepath + "'");
      }
      uri = Uri.parse("file://" + filepath);
    }
    return uri;
  }
  private InputStream getInputStream(String filepath, Context reactContext) throws IORejectionException {
    Uri uri = getFileUri(filepath, false);
    InputStream stream;
    try {
      stream = reactContext.getContentResolver().openInputStream(uri);
    } catch (FileNotFoundException ex) {
      throw new IORejectionException("ENOENT", "ENOENT: " + ex.getMessage() + ", open '" + filepath + "'");
    }
    if (stream == null) {
      throw new IORejectionException("ENOENT", "ENOENT: could not open an input stream for '" + filepath + "'");
    }
    return stream;
  }
  public void AddLineToData(DBScanInput data, String line) {
    if(line!=null) {
      String[] idEmbedding = line.split(",");
      double[] embedding = new double[idEmbedding.length - 1];
      int id = -1;
      for (int i = 0; i < idEmbedding.length; i++) {
        if (i == 0) {
          id = Integer.parseInt((idEmbedding[i]));
        } else {
          embedding[i - 1] = Double.parseDouble(idEmbedding[i]);
        }
      }
      data.ids.add(id);
      data.embedding.add(embedding);
    }
  }

  public  ArrayList<ArrayList<Integer>> ClusterWithFile(String fileUri,float eps, int mintPts,Context context) throws IOException, IORejectionException {
    DBScanInput data = new DBScanInput();
    data.eps = eps;
    data.minPts = mintPts;
    data.embedding = new ArrayList<>();
    data.ids = new ArrayList<>();
    InputStream is;
    BufferedReader reader;
    is = getInputStream(fileUri, context);
    reader = new BufferedReader(new InputStreamReader(is));
    String line = reader.readLine();
    this.AddLineToData(data, line);
    while (line != null) {
      line = reader.readLine();
      this.AddLineToData(data, line);
    }
    List<DoublePoint> embeddings = new ArrayList<>();
    for(int i=0;i<data.embedding.size();i++) {
      embeddings.add(new DoublePoint(data.embedding.get(i)));
    }

    ArrayList<ArrayList<Integer>> allClusters = this.ClusterFaces(data,embeddings);
    List<DoublePoint> nextEmbedding = getNextEmbedding(data,embeddings,allClusters);
    data.eps=data.eps+1;
    ArrayList<ArrayList<Integer>> second = this.ClusterFaces(data,nextEmbedding);
    allClusters.addAll(second);

    data.eps=data.eps+1;
    nextEmbedding = getNextEmbedding(data,embeddings,allClusters);
    ArrayList<ArrayList<Integer>> third = this.ClusterFaces(data,nextEmbedding);
    allClusters.addAll(third);

    data.eps=data.eps+1;
    nextEmbedding = getNextEmbedding(data,embeddings,allClusters);
    ArrayList<ArrayList<Integer>> fourth = this.ClusterFaces(data,nextEmbedding);
    allClusters.addAll(fourth);
    return  allClusters;
  }


  public  List<DoublePoint> getNextEmbedding(DBScanInput data, List<DoublePoint> allEmbeddings, ArrayList<ArrayList<Integer>> currentCluster) {
    ArrayList<Integer> allIds = new ArrayList<>();
    for (ArrayList<Integer> one : currentCluster) {
      allIds.addAll(one);
    }
    List<DoublePoint> embeddings = new ArrayList<>();
    for (int i = 0; i < data.embedding.size(); i++) {
      int id = data.ids.get(i);
      if (!allIds.contains(id)) {
        embeddings.add(allEmbeddings.get(i));
      }
    }
    return  embeddings;
  }
  public ArrayList<ArrayList<Integer>> ClusterFaces(DBScanInput data,List<DoublePoint> embeddings) {
    DistanceMeasure distanceMeasure = new DistanceMeasure() {
      @Override
      public double compute(double[] a, double[] b) throws DimensionMismatchException {
        EuclideanDistance distance = new EuclideanDistance();
        return distance.compute(a,b);
      }
    };

    DBSCANClusterer dbscanClusterer = new DBSCANClusterer(data.eps, data.minPts);
    List<Cluster<DoublePoint>> cluster = dbscanClusterer.cluster(embeddings);
    ArrayList<ArrayList<Integer>> idsClusters = this.getClusterIds(data,cluster);
    return  idsClusters;
  }
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
