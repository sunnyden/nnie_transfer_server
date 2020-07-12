cmake_minimum_required(VERSION 3.15)
project(nnie_transfer_server)

SET(CMAKE_SYSTEM_NAME linux)
SET(CMAKE_SYSTEM_PROCESSOR "arm")
SET(CMAKE_FIND_ROOT_PATH  /opt/hisi-linux/x86-arm/arm-himix200-linux/arm-himix200-linux/)
SET(CMAKE_CROSSCOMPILING true)
SET(CMAKE_C_COMPILER "arm-himix200-linux-gcc")
SET(CMAKE_CXX_COMPILER "arm-himix200-linux-g++")
SET(CMAKE_AR "arm-himix200-linux-ar")
SET(CMAKE_RANLIB "arm-himix200-linux-ranlib")
SET(CMAKE_STRIP "arm-himix200-linux-strip")

SET(TARGET_EXEC nnie_transfer_server)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -mcpu=cortex-a7 -mfloat-abi=softfp -mfpu=neon-vfpv4 -fno-aggressive-loop-optimizations -ldl -ffunction-sections -fdata-sections -O2 -fstack-protector-strong)
include_directories(include)
file(GLOB STATIC_LIBS "./libs/*.a")

add_executable(${TARGET_EXEC}
        src/main.cpp
        src/network/NetworkBase.cpp
        src/network/NetworkBaseThreading.cpp
        src/nnie/mat.cpp
        src/nnie/net.cpp src/nnie/ModelManager.cpp)
target_link_libraries(${TARGET_EXEC} -Wl,--start-group ${STATIC_LIBS} -Wl,--end-group)
target_link_libraries(${TARGET_EXEC} m stdc++ pthread dl rt)