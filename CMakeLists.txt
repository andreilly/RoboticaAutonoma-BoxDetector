cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(tof_image_pallet_box_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug) #RelWithDebInfo


find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost 1.58 REQUIRED COMPONENTS system filesystem iostreams)
include_directories(src ${Boost_INCLUDE_DIRS})

find_package(rofl REQUIRED)
if(${rofl_FOUND})
    include_directories(${rofl_INCLUDE_DIRS})
    link_directories(${rofl_LIBRARY_DIRS})
endif()

add_definitions(-std=c++17)  #PIC in -fPIC = Position Independent Code (basically makes libraries static, default option for SHARED objects/libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

message("rofl_LIBRARIES ${rofl_LIBRARIES}")

set(CONF_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_BINARY_DIR}/include)
configure_file(src/config.h.in ${CMAKE_BINARY_DIR}/include/config.h @ONLY)

add_executable(detector
  apps/detector.cpp    
#   src/ParamMap.cpp #preso da rofl
  src/Utils.cpp
  src/ImgProc.cpp
  
  src/PointCloudPlaneAligner.cpp
  src/BoxRegistration.cpp
  apps/BoxDetector.cpp
#   src/HoughPlaneDetector.cpp
#   src/geometry.cpp
#   src/CurveNNCrust.cpp
#   src/DelaunayTriangulation.cpp
#   src/DisjointSet.cpp
#   src/BoxLayoutMatcher.cpp
#   src/BoxSetConfiguration.cpp
#   src/CurveSimplification.cpp
#   src/PointCurveAccessorVector.cpp
#   src/PolylineCurvatureSimplifier.cpp
)
# target_link_libraries (detector ${PCL_LIBRARIES} ${OpenCV_LIBS})
# message("${PCL_LIBRARIES}")
target_link_libraries (detector ${PCL_LIBRARIES} ${OpenCV_LIBS} ${rofl_LIBRARIES})
#${rofl_LIBRARY})


