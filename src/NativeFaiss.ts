import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type IndexInput = {
  indexFullName: string;
  embedding: number[][];
  dim:number;
  ids: number[];
}
export interface Spec extends TurboModule {
  multiply(a: number, b: number): Promise<number>;
  faissIndex(input:string):Promise<string>;
}


export default TurboModuleRegistry.getEnforcing<Spec>('Faiss');
