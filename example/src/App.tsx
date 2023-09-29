import * as React from 'react';
import * as RNFS from 'react-native-fs';
import { StyleSheet, View, Text } from 'react-native';
import { multiply,faissIndex } from '@cisiwen/react-native-faiss';
import type { IndexInput } from '../../src/NativeFaiss';
import { detectFaces } from '@cisiwen/react-native-photo-ai';
export default function App() {
  const [result, setResult] = React.useState<number | undefined>();
  const [files, setFiles] = React.useState<any[] | undefined>();

   

  let uri = "file:///storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/IMG-20230831-WA0000.jpg";
  let indexInput: IndexInput = {
    embedding: [],
    ids: [],
    dim:512,
    indexFullName:''
  };

 

  React.useEffect(() => {
    multiply(3, 7).then(setResult);
     
    async function test() {
      let indexPath = RNFS.DocumentDirectoryPath + '/faissIndex/';
      await RNFS.mkdir(indexPath);
      indexInput.indexFullName = indexPath + 'facenet.index';
      let faces = await detectFaces(uri);
      faces=JSON.parse(faces.toString());
      faces.forEach((a,i)=>{
        indexInput.ids.push(i);
        indexInput.embedding.push(a.embedding);
      })

      let indexResult= await faissIndex(indexInput);
      console.log(indexResult);
      let dirResult = await RNFS.readDir(indexPath);
      console.log(dirResult);

    }
    test();
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
