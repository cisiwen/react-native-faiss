import { NativeModules, Platform } from 'react-native';
import type { IndexInput, QueryInput, QueryResultItem } from './NativeFaiss';

const LINKING_ERROR =
  `The package 'react-native-faiss' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

// @ts-expect-error
const isTurboModuleEnabled = global.__turboModuleProxy != null;

const FaissModule = isTurboModuleEnabled
  ? require('./NativeFaiss').default
  : NativeModules.Faiss;

const Faiss = FaissModule
  ? FaissModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    );

export function multiply(a: number, b: number): Promise<number> {
  return Faiss.multiply(a, b);
}

export function faissIndex(input:IndexInput): Promise<string> {
  return Faiss.faissIndex(JSON.stringify(input));
}

export async function faissSearch(input:QueryInput): Promise<QueryResultItem[]> {
   let result = await Faiss.queryIndex(JSON.stringify(input));
   if(result){
    return JSON.parse(result);
   }
   else
   {
    let dummy:QueryResultItem[] = [];
    return dummy;
   }
    
}
