#!/bin/bash
cd /media/froduo/4ECEA802CEA7E107/AA_project/16.reId/build-CameraViewer-Desktop-Debug
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../CameraViewer/CameraViewer.pro -spec linux-g++ CONFIG+=debug CONFIG+=qml_debug 2>&1
echo "=== qmake done ==="
make -j4 2>&1
echo "=== make done ==="
