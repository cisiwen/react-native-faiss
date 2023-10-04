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

export type QueryResultItem = {
  id: number;
  score: number;
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
}


export default TurboModuleRegistry.getEnforcing<Spec>('Faiss');
