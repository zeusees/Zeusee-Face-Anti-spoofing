#include <jni.h>
#include <string>
#include "AliveDetector.h"
#include <android/log.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "mytest",__VA_ARGS__)


extern "C" {
JNIEXPORT jint JNICALL
Java_alive_zeusees_activedetection_AliveDetection_detect(JNIEnv *env, jobject obj, jbyteArray yuv,
                                                         jint height, jint width, jlong handle) {
    jbyte *pBuf = (jbyte *) env->GetByteArrayElements(yuv, 0);

    Mat image(height + height / 2, width, CV_8UC1, (unsigned char *) pBuf);

    Mat mBgr;

    cvtColor(image, mBgr, CV_YUV2BGR_NV21);
    transpose(mBgr, mBgr);
    cv::flip(mBgr, mBgr, -1);
    AliveDetector *aliveDetector = (AliveDetector *) handle;
    int state = aliveDetector->detect(mBgr);
    cv::imwrite("/sdcard/test.jpg", mBgr);
    env->ReleaseByteArrayElements(yuv, pBuf, 0);
    return state;
}

std::string jstring2str(JNIEnv *env, jstring jstr) {
    char *rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("GB2312");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte *ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char *) malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    std::string stemp(rtn);
    free(rtn);
    return stemp;
}

JNIEXPORT jlong JNICALL
Java_alive_zeusees_activedetection_AliveDetection_init(JNIEnv *env, jobject obj, jstring folder) {
    std::string detector_path = jstring2str(env, folder);
    AliveDetector *aliveDetector = new AliveDetector(detector_path);
    return (jlong) aliveDetector;

    //    return 0;

}

JNIEXPORT void JNICALL
Java_alive_zeusees_activedetection_AliveDetection_release(JNIEnv *env, jobject obj, jlong handle) {
    AliveDetector *PR = (AliveDetector *) handle;
    delete PR;
}

}


