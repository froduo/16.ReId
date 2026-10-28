#!/bin/bash
cd /media/froduo/4ECEA802CEA7E107/AA_project/16.reId
echo "Testing RTSP stream..."
timeout 10 ffprobe -v error -rtsp_transport tcp -i "rtsp://192.168.99.10:8554/live/camera" -show_entries stream=codec_name,width,height -of json 2>&1
echo "---"
echo "Exit code: $?"
echo "---"
echo "Testing HTTP API..."
curl -s --connect-timeout 5 -u admin:hf123456 -X POST 'http://192.168.99.10/ImageAndVideo/CapturePictureBase64' 2>&1 | head -200
echo "---"
echo "Testing RTMP pull settings API..."
curl -s --connect-timeout 5 -u admin:hf123456 'http://192.168.99.10/ImageAndVideo/RtmpPullGet' 2>&1 | head -200
echo "---"
echo "Done"
