cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

# OJITO el directorio cambia por cada instalacion eso deben cambiarlo
set(OpenCV_DIR C:/Users/lesly.fuentes/Desktop/U/PDI/AyudantiaPDI2022-2-main/AyudantiaPDI2022-2-main/media/opencv)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main main.cpp)

target_link_libraries( main ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)