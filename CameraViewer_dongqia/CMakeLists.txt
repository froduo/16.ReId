cmake_minimum_required(VERSION 3.16)
project(CameraViewer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt 配置
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 查找 Qt6 或 Qt5
find_package(Qt6 COMPONENTS Widgets QUIET)
if(NOT Qt6_FOUND)
    find_package(Qt5 5.15 REQUIRED COMPONENTS Widgets)
    message(STATUS "Using Qt5: ${Qt5_VERSION}")
else()
    message(STATUS "Using Qt6: ${Qt6_VERSION}")
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# 源文件
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/camera_thread.cpp
    src/camera_thread.h
)

set(HEADERS
    src/mainwindow.h
    src/camera_thread.h
)

set(UI_FILES
    src/mainwindow.ui
)

# 创建可执行文件
if(Qt6_FOUND)
    qt_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets ${OpenCV_LIBS})
else()
    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets ${OpenCV_LIBS})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# 安装
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
