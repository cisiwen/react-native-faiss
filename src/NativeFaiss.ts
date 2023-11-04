import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type IndexInput = {
  indexFullName: string;
  embedding: number[][];
  dim:number;
  ids: number[];
  template?:number[];
  ended:string;
}

export type QueryInput = {
  indexFullName: string;
  queryVector: number[];
  k: number;
}

export type ClusteringInput = {
  startOfData:boolean;
  endOfData:boolean;
  eps:number;
  minPts:number;
  embedding: number[][];
  ids: number[];
}

export type ClusteringOutput = {
  totalData:number;
  clusters:any[]
}

export type QueryResultItem = {
  id: number;
  score: number;
}
export type KmeanClusterItem = {
  id: number;
  clusterId:number;
}

export type TrainIndexInput = {
  embedding: number[][];
  dim:number;
}

export interface Spec extends TurboModule {
  multiply(a: number, b: number): Promise<number>;
  faissIndex(input:string):Promise<string>;
  queryIndex(input:string):Promise<string>;
  trainIndex(input:string):Promise<string>;
  dbscanClustering(input:string):Promise<ClusteringOutput>;
  clusterWithFile(fileUri:string,eps:number,minPts:number):Promise<ClusteringOutput>;
  kmeansCluster(fileUri:string, k:number, dim:number,size:number):Promise<KmeanClusterItem[]>;
}


export default TurboModuleRegistry.getEnforcing<Spec>('Faiss');
