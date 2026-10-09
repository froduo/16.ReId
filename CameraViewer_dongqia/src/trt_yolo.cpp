#include "trt_yolo.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/dnn.hpp>

using namespace nvinfer1;

TrtYolo::TrtYolo(const std::string& enginePath, ILogger& logger, float confThresh)
    : logger(logger), confThreshold(confThresh)
{
    buffers[0] = nullptr;
    buffers[1] = nullptr;

    if (!loadEngine(enginePath)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + enginePath);
    }
}

TrtYolo::~TrtYolo()
{
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    if (buffers[0]) cudaFree(buffers[0]);
    if (buffers[1]) cudaFree(buffers[1]);

    cudaStreamDestroy(stream);
}

size_t TrtYolo::getSizeByDim(const Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}

bool TrtYolo::loadEngine(const std::string& enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[TrtYolo] Cannot open engine file: " << enginePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 验证是否为 TensorRT 引擎
    if (size >= 4) {
        const char* magic = engineData.data();
        bool isTrtEngine = (magic[0] == 'f' && magic[1] == 't' && magic[2] == 'r' && magic[3] == 't');
        if (!isTrtEngine) {
            std::cerr << "[TrtYolo] File is not a valid TensorRT engine: " << enginePath << std::endl;
            return false;
        }
    }

    runtime = createInferRuntime(logger);
    if (!runtime) return false;

    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine) return false;

    context = engine->createExecutionContext();
    if (!context) return false;

    // 扫描所有 IO tensor
    int nbIOTensors = engine->getNbIOTensors();
    int inputIdx = -1, outputIdx = -1;

    for (int i = 0; i < nbIOTensors; i++)
    {
        const char* tensorName = engine->getIOTensorName(i);
        TensorIOMode mode = engine->getTensorIOMode(tensorName);

        if (mode == TensorIOMode::kINPUT) {
            inputIdx = i;
            inputTensorName = tensorName;
        } else {
            if (outputIdx < 0) {
                outputIdx = i;
                outputTensorName = tensorName;
            }
        }
    }

    if (inputIdx < 0 || outputIdx < 0) {
        std::cerr << "[TrtYolo] Cannot find input/output tensors" << std::endl;
        return false;
    }

    // 获取输入尺寸
    auto inputDims = engine->getTensorShape(inputTensorName.c_str());
    if (inputDims.nbDims >= 4) {
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
    }
    context->setInputShape(inputTensorName.c_str(), Dims4(1, 3, inputH, inputW));

    inputSize = 1 * 3 * inputH * inputW * sizeof(float);

    // 获取输出尺寸
    auto outDims = engine->getTensorShape(outputTensorName.c_str());
    outputSize = getSizeByDim(outDims) * sizeof(float);

    hostInput.resize(inputSize / sizeof(float));
    hostOutput.resize(outputSize / sizeof(float));

    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize);

    cudaStreamCreate(&stream);

    std::cout << "[TrtYolo] Engine loaded: " << enginePath
              << " input=" << inputW << "x" << inputH
              << " output_size=" << hostOutput.size() << std::endl;
    return true;
}

void TrtYolo::preprocess(const cv::Mat& img)
{
    lastImage = img;
    int w = img.cols;
    int h = img.rows;
    scale = std::min((float)inputW / w, (float)inputH / h);
    int newW = int(w * scale);
    int newH = int(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));

    padX = (inputW - newW) / 2.0f;
    padY = (inputH - newH) / 2.0f;

    cv::Mat padded(inputH, inputW, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(padded(cv::Rect(static_cast<int>(padX), static_cast<int>(padY), newW, newH)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW
    const int planeSize = inputH * inputW;
    const float* src = reinterpret_cast<const float*>(floatImg.data);
    for (int i = 0; i < planeSize; ++i) {
        const int base = i * 3;
        hostInput[i] = src[base + 0];
        hostInput[planeSize + i] = src[base + 1];
        hostInput[2 * planeSize + i] = src[base + 2];
    }

    cudaMemcpyAsync(buffers[0], hostInput.data(), inputSize, cudaMemcpyHostToDevice, stream);
}

void TrtYolo::infer()
{
    context->setTensorAddress(inputTensorName.c_str(), buffers[0]);
    context->setTensorAddress(outputTensorName.c_str(), buffers[1]);

    context->enqueueV3(stream);

    cudaMemcpyAsync(hostOutput.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void TrtYolo::postprocess(std::vector<Detection>& results)
{
    results.clear();

    auto outDims = engine->getTensorShape(outputTensorName.c_str());
    if (outDims.nbDims < 3) return;

    const int dim1 = outDims.d[1];
    const int dim2 = outDims.d[2];

    // 诊断：记录原始输出维度
    static int s_postLogCount = 0;
    if (s_postLogCount < 3) {
        qDebug() << "[TrtYolo] postprocess: output dims =" << outDims.nbDims
                 << "dim1=" << dim1 << "dim2=" << dim2
                 << "confThreshold=" << confThreshold;
    }

    int totalAboveThresh = 0;   // 超过置信度阈值的总检测数
    int personDetections = 0;   // person 类别检测数
    int bestNonPersonClass = -1;
    float bestNonPersonScore = 0.0f;

    // yolo26 格式 (NMS 预过滤): [1, numDet, 6]
    if (dim1 <= 300 && dim2 == 6) {
        int numDet = dim1;
        for (int i = 0; i < numDet; i++) {
            float conf = hostOutput[i * 6 + 4];
            if (conf < confThreshold) continue;
            int classId = static_cast<int>(hostOutput[i * 6 + 5]);
            totalAboveThresh++;

            // 记录最高分非 person 检测
            if (classId != 0 && conf > bestNonPersonScore) {
                bestNonPersonScore = conf;
                bestNonPersonClass = classId;
            }

            // 仅保留 person 类别 (class_id=0)
            if (classId != 0) continue;
            personDetections++;

            float x1 = (hostOutput[i * 6 + 0] - padX) / scale;
            float y1 = (hostOutput[i * 6 + 1] - padY) / scale;
            float x2 = (hostOutput[i * 6 + 2] - padX) / scale;
            float y2 = (hostOutput[i * 6 + 3] - padY) / scale;

            Detection det;
            det.x = std::max(0, (int)x1);
            det.y = std::max(0, (int)y1);
            det.w = std::min((int)(x2 - x1), lastImage.cols - det.x);
            det.h = std::min((int)(y2 - y1), lastImage.rows - det.y);
            det.score = conf;
            det.class_id = classId;

            if (det.w > 0 && det.h > 0)
                results.push_back(det);
        }

        // 诊断日志
        if (s_postLogCount < 3) {
            qDebug() << "[TrtYolo] yolo26 format: totalAboveThresh=" << totalAboveThresh
                     << "personDetections=" << personDetections
                     << "bestNonPersonClass=" << bestNonPersonClass
                     << "bestNonPersonScore=" << bestNonPersonScore;
            s_postLogCount++;
        }
        return;
    }

    // yolo12 格式 (稠密网格): [1, numAttr, numPred] 或 [1, numPred, numAttr]
    bool isLayoutA = (dim1 >= 5 && dim1 <= 512 && dim2 > dim1);
    int numAttr = isLayoutA ? dim1 : dim2;
    int numPred = isLayoutA ? dim2 : dim1;
    int numClasses = numAttr - 4;
    if (numClasses <= 0) return;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < numPred; i++)
    {
        // 仅检测 person 类别 (class 0)
        float personScore = isLayoutA ? hostOutput[4 * numPred + i] : hostOutput[i * numAttr + 4];
        if (personScore < confThreshold) continue;

        float cx, cy, w, h;
        if (isLayoutA) {
            cx = hostOutput[0 * numPred + i]; cy = hostOutput[1 * numPred + i];
            w  = hostOutput[2 * numPred + i]; h  = hostOutput[3 * numPred + i];
        } else {
            cx = hostOutput[i * numAttr + 0]; cy = hostOutput[i * numAttr + 1];
            w  = hostOutput[i * numAttr + 2]; h  = hostOutput[i * numAttr + 3];
        }

        cx = (cx - padX) / scale; cy = (cy - padY) / scale;
        w  = w / scale;  h  = h / scale;

        int x = std::max(0, int(cx - w / 2));
        int y = std::max(0, int(cy - h / 2));
        int iw = std::min((int)w, lastImage.cols - x);
        int ih = std::min((int)h, lastImage.rows - y);
        if (iw <= 0 || ih <= 0) continue;

        boxes.emplace_back(x, y, iw, ih);
        scores.push_back(personScore);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, 0.45f, indices);

    for (int idx : indices) {
        Detection det;
        det.x = boxes[idx].x; det.y = boxes[idx].y;
        det.w = boxes[idx].width; det.h = boxes[idx].height;
        det.score = scores[idx]; det.class_id = 0;
        results.push_back(det);
    }

    // 诊断日志
    if (s_postLogCount < 3) {
        qDebug() << "[TrtYolo] yolo12 format: boxes before NMS=" << boxes.size()
                 << "after NMS=" << results.size()
                 << "numPred=" << numPred
                 << "numClasses=" << numClasses
                 << "confThreshold=" << confThreshold;
        s_postLogCount++;
    }
}

// ===== postprocessAll: 返回所有类别检测结果（不过滤 class_id，用于诊断）=====
void TrtYolo::postprocessAll(std::vector<Detection>& results, float lowThreshold)
{
    results.clear();

    auto outDims = engine->getTensorShape(outputTensorName.c_str());
    if (outDims.nbDims < 3) return;

    const int dim1 = outDims.d[1];
    const int dim2 = outDims.d[2];

    // yolo26 格式 [1, numDet, 6]
    if (dim1 <= 300 && dim2 == 6) {
        int numDet = dim1;
        for (int i = 0; i < numDet; i++) {
            float conf = hostOutput[i * 6 + 4];
            if (conf < lowThreshold) continue;
            int classId = static_cast<int>(hostOutput[i * 6 + 5]);

            float x1 = (hostOutput[i * 6 + 0] - padX) / scale;
            float y1 = (hostOutput[i * 6 + 1] - padY) / scale;
            float x2 = (hostOutput[i * 6 + 2] - padX) / scale;
            float y2 = (hostOutput[i * 6 + 3] - padY) / scale;

            Detection det;
            det.x = std::max(0, (int)x1);
            det.y = std::max(0, (int)y1);
            det.w = std::min((int)(x2 - x1), lastImage.cols - det.x);
            det.h = std::min((int)(y2 - y1), lastImage.rows - det.y);
            det.score = conf;
            det.class_id = classId;
            if (det.w > 0 && det.h > 0)
                results.push_back(det);
        }
        return;
    }

    // yolo12 格式
    bool isLayoutA = (dim1 >= 5 && dim1 <= 512 && dim2 > dim1);
    int numAttr = isLayoutA ? dim1 : dim2;
    int numPred = isLayoutA ? dim2 : dim1;
    int numClasses = numAttr - 4;
    if (numClasses <= 0) return;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int i = 0; i < numPred; i++)
    {
        // 找最高分类别
        float bestScore = 0;
        int bestClass = 0;
        for (int c = 0; c < numClasses; c++) {
            float s = isLayoutA ? hostOutput[(4 + c) * numPred + i] : hostOutput[i * numAttr + 4 + c];
            if (s > bestScore) { bestScore = s; bestClass = c; }
        }
        if (bestScore < lowThreshold) continue;

        float cx, cy, w, h;
        if (isLayoutA) {
            cx = hostOutput[0 * numPred + i]; cy = hostOutput[1 * numPred + i];
            w  = hostOutput[2 * numPred + i]; h  = hostOutput[3 * numPred + i];
        } else {
            cx = hostOutput[i * numAttr + 0]; cy = hostOutput[i * numAttr + 1];
            w  = hostOutput[i * numAttr + 2]; h  = hostOutput[i * numAttr + 3];
        }

        cx = (cx - padX) / scale; cy = (cy - padY) / scale;
        w  = w / scale;  h  = h / scale;

        int x = std::max(0, int(cx - w / 2));
        int y = std::max(0, int(cy - h / 2));
        int iw = std::min((int)w, lastImage.cols - x);
        int ih = std::min((int)h, lastImage.rows - y);
        if (iw <= 0 || ih <= 0) continue;

        boxes.emplace_back(x, y, iw, ih);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, lowThreshold, 0.45f, indices);

    for (int idx : indices) {
        Detection det;
        det.x = boxes[idx].x; det.y = boxes[idx].y;
        det.w = boxes[idx].width; det.h = boxes[idx].height;
        det.score = scores[idx]; det.class_id = classIds[idx];
        results.push_back(det);
    }
}
