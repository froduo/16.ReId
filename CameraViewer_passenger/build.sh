#!/bin/bash
cd build-CameraViewer-Desktop-Debug
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../CameraViewer/CameraViewer.pro -spec linux-g++ CONFIG+=debug CONFIG+=qml_debug
make -j4
echo "BUILD_EXIT_CODE=$?"
