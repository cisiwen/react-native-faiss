#include <jni.h>
#include<array>
#include <string>
#include<iostream>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>

#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "log.h"
#include "faiss/AutoTune.h"
#include "MetaIndexes.h"

int64_t getCurrentMillTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t) tv.tv_sec * 1000 + (int64_t) tv.tv_usec / 1000);//毫秒
}


#define FEATURE_COUNT 512

const std::string L2 = "l2";
const std::string INNER_PRODUCT = "innerproduct";

std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) {
    if (javaString == nullptr) {
        throw std::runtime_error("String cannot be null");
    }

    const char *cString = env->GetStringUTFChars(javaString, nullptr);
    if (cString == nullptr) {

        // Will only reach here if there is no exception in the stack, but the call failed
        throw std::runtime_error("Unable to convert java string to cpp string");
    }
    std::string cppString(cString);
    env->ReleaseStringUTFChars(javaString, cString);
    return cppString;
}
std::vector<int64_t>  ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int numElements = env->GetArrayLength(arrayJ);
    int *arrayCpp = env->GetIntArrayElements(arrayJ, nullptr);
    if (arrayCpp == nullptr) {
        throw std::runtime_error("Unable to get integer array elements");
    }

    std::vector<int64_t> vectorCpp;
    vectorCpp.reserve(numElements);
    for (int i = 0; i < numElements; ++i) {
        vectorCpp.push_back(arrayCpp[i]);
    }
    env->ReleaseIntArrayElements(arrayJ, arrayCpp, JNI_ABORT);
    return vectorCpp;
}

std::vector<float> Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ,
                                                                              int dim) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int numVectors = env->GetArrayLength(array2dJ);
    //this->HasExceptionInStack(env);

    std::vector<float> floatVectorCpp;
    for (int i = 0; i < numVectors; ++i) {
        auto vectorArray = (jfloatArray)env->GetObjectArrayElement(array2dJ, i);
        //this->HasExceptionInStack(env, "Unable to get object array element");

        if (dim != env->GetArrayLength(vectorArray)) {
            throw std::runtime_error("Dimension of vectors is inconsistent");
        }

        float* vector = env->GetFloatArrayElements(vectorArray, nullptr);
        if (vector == nullptr) {
            //this->HasExceptionInStack(env);
            throw std::runtime_error("Unable to get float array elements");
        }

        for(int j = 0; j < dim; ++j) {
            floatVectorCpp.push_back(vector[j]);
        }
        env->ReleaseFloatArrayElements(vectorArray, vector, JNI_ABORT);
    }
    //this->HasExceptionInStack(env);
    env->DeleteLocalRef(array2dJ);
    return floatVectorCpp;
}

jfloat * GetFloatArrayElements(JNIEnv *env, jfloatArray array, jboolean * isCopy) {
    float* floatArray = env->GetFloatArrayElements(array, nullptr);
    if (floatArray == nullptr) {
        throw std::runtime_error("Unable to get float elements");
    }

    return floatArray;
}


void ReleaseFloatArrayElements(JNIEnv *env, jfloatArray array, jfloat *elems, int mode) {
    env->ReleaseFloatArrayElements(array, elems, mode);
}

faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType) {
    if (spaceType == L2) {
        return faiss::METRIC_L2;
    }

    if (spaceType == INNER_PRODUCT) {
        return faiss::METRIC_INNER_PRODUCT;
    }

    throw std::runtime_error("Invalid spaceType");
}

jlong LoadIndex(JNIEnv * env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    faiss::Index* indexReader = faiss::read_index(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY);
    return (jlong) indexReader;
}

jobjectArray NewObjectArray(JNIEnv *env, jsize len, jclass clazz, jobject init) {
    jobjectArray objectArray = env->NewObjectArray(len, clazz, init);
    if (objectArray == nullptr) {
        throw std::runtime_error("Unable to allocate object array");
    }

    return objectArray;
}


void SetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index, jobject val) {
    env->SetObjectArrayElement(array, index, val);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_faiss_FaissManager_indexFromAndroid(JNIEnv *env, jclass clazz,
                                                         jobjectArray embedding, jintArray ids,jint dim,jstring indexName) {
    //faiss::IndexFlatL2 *index = new faiss::IndexFlatL2(dim);
    std::string name_faiss = ConvertJavaStringToCppString(env,indexName);
    std::string indexDescriptionCpp="Flat";
    std::string  result = "good";
    std::string  spaceTypeCpp = "l2";
    int length = env->GetArrayLength(embedding);
    int idsLength = env->GetArrayLength(ids);
    auto vectors = Convert2dJavaObjectArrayToCppFloatVector(env,embedding,dim);
    auto idVector = ConvertJavaIntArrayToCppIntVector(env, ids);

    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::index_factory(dim, indexDescriptionCpp.c_str(),metric));

    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    faiss::IndexIDMap idMap = faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(length, vectors.data(), idVector.data());
    faiss::write_index(&idMap, name_faiss.c_str());
    //delete indexWriter;
    return env->NewStringUTF(std::to_string(vectors.size()).c_str());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_faiss_FaissManager_QueryIndex(JNIEnv * env,  jclass clazz,jstring indexPathJ, jfloatArray queryVectorJ, jint kJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    faiss::Index* indexReader = faiss::read_index(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    // The ids vector will hold the top k ids from the search and the dis vector will hold the top k distances from
    // the query point
    std::vector<float> dis(kJ);
    std::vector<int64_t> ids(kJ);
    float* rawQueryvector = GetFloatArrayElements(env, queryVectorJ, nullptr);
    /*
        Setting the omp_set_num_threads to 1 to make sure that no new OMP threads are getting created.
    */
    //omp_set_num_threads(1);
    // create the filterSearch params if the filterIdsJ is not a null pointer

    try {
        indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data());
    } catch (...) {
        ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
        throw;
    }

    ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);

    // If there are not k results, the results will be padded with -1. Find the first -1, and set result size to that
    // index
    int resultSize = kJ;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }


    jclass resultClass = env->FindClass("com/faiss/models/KNNQueryResult");
    jmethodID allArgs = env->GetMethodID(resultClass, "<init>", "(IF)V");
    jobjectArray results = NewObjectArray(env, resultSize, resultClass, nullptr);

    jobject result;
    for(int i = 0; i < resultSize; ++i) {
        result = env->NewObject(resultClass, allArgs, ids[i], dis[i]);
        SetObjectArrayElement(env, results, i, result);
    }
    return results;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_faiss_FaissManager_stringFromJNI(JNIEnv *env, jclass clazz, jint number) {
    std::string result = "0";
    std::string name_faiss = "/sdcard/tmp/1.index";
    faiss::IndexFlatL2 *index;

    index = new faiss::IndexFlatL2(FEATURE_COUNT);
    float data[FEATURE_COUNT];//random data
    for (int i = 0; i < FEATURE_COUNT; i++) {
        data[i] = 0.0001f * i + 0.00001f;
    }
    LOGI("index->add。。。");
    for (int i = 0; i < (1000000); i++) {

        index->add(1, data);
        //Problems may occur in 32-bit
        LOGD("SIZE= %lld", index->ntotal * index->d);
    }
    LOGI("index->add over");


    LOGI("save...");
    //please apply for sdcard permission before writing
    faiss::write_index(index, name_faiss.c_str());
    LOGI("save ");
    delete index;


    LOGI("read ... ");
    faiss::Index *tmp = faiss::read_index(name_faiss.c_str(), faiss::IO_FLAG_MMAP);
    LOGI("read ok");
    //null point check
    index = (faiss::IndexFlatL2 *) tmp;

    LOGI("index->d=%d", index->d);


    float read[FEATURE_COUNT] = {0};
    index->reconstruct(index->ntotal - 1, read);

    for (int i = 0; i < FEATURE_COUNT; i++) {
        LOGI("read[%d] %f %f", i, data[i], read[i]);
    }

    float data2[FEATURE_COUNT];//random data
    for (int i = 0; i < FEATURE_COUNT; i++) {
        data2[i] = 0.0001f * FEATURE_COUNT * 2 + 0.00002f;
    }
    const int64_t destCount = 3;
    int64_t *listIndex = (int64_t *) malloc(sizeof(int64_t) * destCount);
    float *listScore = (float *) malloc(sizeof(float) * destCount);
    LOGI("index->search。。。");
    index->search(1, data2, destCount, listScore, listIndex);
    LOGI("index->search");
    for (int i = 0; i < destCount; i++) {
        LOGI("index->search[%lld]=%f", listIndex[i], listScore[i]);
    }
    free(listIndex);
    free(listScore);

    LOGI("read");
    result = std::to_string(index->ntotal);
    return env->NewStringUTF(result.c_str());
}


