import * as React from 'react';
import * as RNFS from 'react-native-fs';
import { StyleSheet, View, Text } from 'react-native';
import { multiply,faissIndex } from 'react-native-faiss';
import type { IndexInput } from '../../src/NativeFaiss';

export default function App() {
  const [result, setResult] = React.useState<number | undefined>();
  const [files, setFiles] = React.useState<any[] | undefined>();

   

  let indexInput: IndexInput = {
    embedding: [],
    ids: [],
    dim:512,
    indexFullName:''
  };

  let size=10;
  for(let i=0;i<size;i++){
    indexInput.ids.push(i);
    let embedding=[];
    for(let j=0;j<indexInput.dim;j++){
      embedding.push(Math.random());
    }
    indexInput.embedding.push(embedding);
  }

  React.useEffect(() => {
    multiply(3, 7).then(setResult);
    let indexPath = RNFS.DocumentDirectoryPath + '/faissIndex/';
    RNFS.mkdir(indexPath).then(() => {
       indexInput.indexFullName = indexPath + 'facenet.index';
        faissIndex(indexInput).then((res) => {
          console.log(res);
        });
    });
 
    RNFS.readDir(indexPath).then((res) => {
      res.forEach((file) => {
        console.log(file);
      })
    })
  }, []);

  return (
    <View style={styles.container}>
      <Text>Result: {result}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  box: {
    width: 60,
    height: 60,
    marginVertical: 20,
  },
});
