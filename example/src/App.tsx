import * as React from 'react';
import * as RNFS from 'react-native-fs';
import { StyleSheet, View, Text, Image, ScrollView, FlatList } from 'react-native';
import { multiply, faissIndex, faissSearch, trainIndex, dbscanClustering } from '../../src/index';
import type { ClusteringInput, IndexInput, QueryInput, TrainIndexInput } from '../../src/NativeFaiss';
import { detectFaces } from '@cisiwen/react-native-photo-ai';
import datas from './data.json';
import type { FaceDetectionResult } from '@cisiwen/react-native-photo-ai/lib/typescript/NativePhotoAi';
export default function App() {
  const [result, setResult] = React.useState<number | undefined>();
  const [files, setFiles] = React.useState<any[] | undefined>();
  let [faces , setFaces] = React.useState<{id: number, face: FaceDetectionResult[] }[]>([]);


  let uri = "file:///storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/IMG-20230831-WA0000.jpg";
  let indexInput: IndexInput = {
    embedding: [],
    ids: [],
    dim: 512,
    indexFullName: '',
    ended: '1'
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

    async function test(url: string, id: number, ended: boolean = false) {
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


      indexInput.ended = ended ? "1" : "0";
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
          console.log("faissSearch", result);
        }
      }
    }


   
    const facenetDetect = async () => {
      let clusteringInputFile = RNFS.DocumentDirectoryPath + '/clusteringInput.json';
      let imageFacesFile = RNFS.DocumentDirectoryPath + '/imageFaces.json';

      if (await RNFS.exists(clusteringInputFile) && await RNFS.exists(imageFacesFile)) {
        let clusteringInputs: ClusteringInput[] = JSON.parse(await RNFS.readFile(clusteringInputFile));
        let downloadClusteOutput = `${RNFS.DownloadDirectoryPath}/clusteringOutput.json`;
        await RNFS.writeFile(downloadClusteOutput, JSON.stringify(clusteringInputs));
        //clusteringInputs=clusteringInputs.splice(0,10);
        console.log("clusteringInputs", clusteringInputs.length);
        let imageFaces: { [key: string]: { url: string, faces: { id: number, face: FaceDetectionResult }[] } } = JSON.parse(await RNFS.readFile(imageFacesFile));
        console.log("imageFaces", imageFaces != null);
        for (let i = 0; i < clusteringInputs.length; i++) {
          let clusteringInput = clusteringInputs[i];
          if (clusteringInput) {
            clusteringInput.eps = 0.34;
            clusteringInput.minPts = 2;
            clusteringInput.startOfData = i == 0;
            clusteringInput.endOfData = i == clusteringInputs.length - 1;
            let result = await dbscanClustering(clusteringInput);
            if (result.clusters != undefined) {
              console.log("dbscanClustering cluster result", i,result.clusters.length, result.clusters);
              let faces: { id: number, face: FaceDetectionResult[] }[] = [];

              result.clusters.forEach((a,i) => {
                let group = faces.find((b) => b.id == i);
                if (group == undefined) {
                  group = {
                    id: i,
                    face: []
                  }
                  faces.push(group);
                }
                a.forEach((b:number) => {
                   for(var f in imageFaces){
                     let face = imageFaces[f]?.faces.find((c) => c.id == b);
                     if(face){
                       group?.face.push(face.face);
                     }
                   }
                })
              })
              setFaces(faces);
            }
            else {
              console.log("dbscanClustering added", i, result);
            }
          }
        }

      }
      else {
        let imagesData = datas.filter((a) => a.includes("jpg") || a.includes("png") || a.includes("heic"));
        let imageFaces: {
          [key: string]: {
            url: string,
            faces: { id: number, face: FaceDetectionResult }[]
          }
        } = {}

        let allInput: ClusteringInput[] = [];
        let lastClusterInput: ClusteringInput | undefined;
        let faceId: number = 1;
        let isSTART: boolean = true;
        for (let i = 0; i < 200; i++) {
          let url: string | undefined = imagesData[i];
          if (url != undefined) {
            let faces = await detectFaces(url, "testing", imagesPath);
            imageFaces[i.toString()] = {
              url: url,
              faces: []
            }
            let clusteringInput: ClusteringInput = {
              startOfData: i == 0,
              endOfData: false,
              eps: 0.123,
              minPts: 0,
              embedding: [],
              ids: []
            }
            faces.forEach((a, idx) => {
              faceId++;
              console.log("facenetDetect", i, url, idx, a.embedding.length);
              imageFaces[i.toString()]?.faces.push({
                id: faceId,
                face: a
              });
              clusteringInput.embedding.push(a.embedding);
              clusteringInput.ids.push(faceId);
            });
            if (i >= 199) {
              lastClusterInput = clusteringInput;
            }
            else {
              if (clusteringInput.embedding.length > 0) {
                clusteringInput.startOfData = isSTART;
                let result = await dbscanClustering(clusteringInput);
                isSTART = false;
                console.log("dbscanClustering", i, result);
                allInput.push(clusteringInput);
              }

            }
          }
        }
        if (lastClusterInput) {
          lastClusterInput.endOfData = true;
          let result = await dbscanClustering(lastClusterInput);
          console.log("dbscanClustering", result);
          allInput.push(lastClusterInput);
        }
        await RNFS.writeFile(clusteringInputFile, JSON.stringify(allInput));
        await RNFS.writeFile(imageFacesFile, JSON.stringify(imageFaces));
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
      await test(uri, 100, false);
      await test(url2, 10000, false);
      let dirResult = await RNFS.readDir(indexPath);
      console.log(dirResult[0]);
    }

    const startTrainIndex = async () => {
      let imagesData = datas.filter((a) => a.includes("jpg") || a.includes("png") || a.includes("heic"));
      let trainInput: TrainIndexInput = {
        dim: 512,
        embedding: [],
      };

      let i = 0;
      while (trainInput.embedding.length < 5) {
        let url = imagesData.pop()
        console.log("train", i, url, trainInput.embedding.length);
        if (url) {
          let faces = await detectFaces(url, "testing", imagesPath);
          console.log("train faces", i, url, faces.length);
          faces.forEach((a, idx) => {
            trainInput.embedding.push(a.embedding);
          })
        }
        i++;
      }

      console.log("trainstart", trainInput.embedding.length);
      let result = await trainIndex(trainInput);
      console.log(result);
    }

    const indexWithTemplate = async () => {
      let template = [73, 120, 70, 50, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    //startTrainIndex();
    ///indexAll();
    //search();
    facenetDetect();
  }, []);
  const renderGrid = (data:{id: number, face: FaceDetectionResult[] },rootIndex:number) => {
    return (
      <FlatList
      horizontal={true}
      keyExtractor={(item,index)=>`${index}_${data.id}_${rootIndex}}`}
      renderItem={(item)=>{
        return <Image  style={{width:100,height:100}} source={{uri:`file:///${item.item.url}`}}></Image>
      }}
      data={data.face}
       />
    )
  }
  return (
    <ScrollView style={styles.container}>
      
      {
        faces.map((a,index) => {
          return renderGrid(a,index);
        })
      }
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    
  },
  box: {
    width: 60,
    height: 60,
    marginVertical: 20,
  },
});
