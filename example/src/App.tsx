import * as React from 'react';
import * as RNFS from 'react-native-fs';
import { StyleSheet, View, Text } from 'react-native';
import { multiply, faissIndex,faissSearch,trainIndex} from '../../src/index';
import type { IndexInput, QueryInput, TrainIndexInput } from '../../src/NativeFaiss';
import { detectFaces } from '@cisiwen/react-native-photo-ai';
import datas from './data.json';
export default function App() {
  const [result, setResult] = React.useState<number | undefined>();
  const [files, setFiles] = React.useState<any[] | undefined>();



  let uri = "file:///storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/IMG-20230831-WA0000.jpg";
  let indexInput: IndexInput = {
    embedding: [],
    ids: [],
    dim: 512,
    indexFullName: ''
  };


  let url2 = "file:///storage/emulated/0/DCIM/Camera/20230925_084521.heic"

  React.useEffect(() => {
    multiply(3, 7).then(setResult);

    let indexPath = RNFS.DocumentDirectoryPath + '/faissIndex/';
    let imagesPath = RNFS.DocumentDirectoryPath + '/images/';
    indexInput.indexFullName = indexPath + 'facenet.index';

    async function propare() {
      await RNFS.mkdir(indexPath);
      await RNFS.mkdir(imagesPath);
    }

    async function test(url: string, id: number,ended:boolean=false) {
      console.log("test", url, id);

      let faces = await detectFaces(url, "testing", imagesPath);
      indexInput.embedding = [];
      indexInput.ids = [];
      //indexInput.template=[73,120,70,50,0,2,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,16,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0];
      console.log(faces);
      faces.forEach((a, i) => {
        indexInput.ids.push(i + id);
        indexInput.embedding.push(a.embedding);
      })
      

      indexInput.ended = ended  ? "1" :"0";
      let indexResult = await faissIndex(indexInput);
      console.log(indexResult);
      let dirResult = await RNFS.readDir(indexPath);
      console.log(dirResult[0]);

    }


    const search = async () => {
      let faces = await detectFaces(url2, "testing", imagesPath);
      console.log(faces);
      for (let i = 0; i < faces.length; i++) {
        let face = faces[i];
        if (face) {
          let queryIndex: QueryInput = {
            queryVector: face?.embedding,
            indexFullName: indexInput?.indexFullName ?? '',
            k: 10
          }
          let result = await faissSearch(queryIndex);
          console.log("faissSearch",result);
        }
      }
    }
    //search();

    async function indexAll() {
      //await propare();
      let imagesData = datas.filter((a) => a.includes("jpg") || a.includes("png") || a.includes("heic"));
      for (let i = 0; i < 200; i++) {
        let url: string | undefined = imagesData[i];
        if (url != undefined) {
          await test(url, i);
        }
      }
      await test(uri, 100,false);
      await test(url2, 10000,false);
      let dirResult = await RNFS.readDir(indexPath);
      console.log(dirResult[0]);
    }

    const startTrainIndex = async()=>{
      let imagesData = datas.filter((a) => a.includes("jpg") || a.includes("png") || a.includes("heic"));
      let trainInput:TrainIndexInput = {
        dim:512,
        embedding:[],
      };

      let i = 0;
      while(trainInput.embedding.length<5){
        let url = imagesData.pop()
        console.log("train",i,url,trainInput.embedding.length);
        if(url){
          let faces = await detectFaces(url, "testing", imagesPath);
          console.log("train faces",i,url,faces.length);
          faces.forEach((a,idx) => {
            trainInput.embedding.push(a.embedding);
          })
        }
        i++;
      }

      console.log("trainstart",trainInput.embedding.length);
      let result = await trainIndex(trainInput);
      console.log(result);
    }

    const indexWithTemplate = async()=>{
      let template=[73,120,70,50,0,2,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,16,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0];
    }
    //startTrainIndex();
    ///indexAll();
    search();
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
