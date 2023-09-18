cmake_minimum_required(VERSION 3.0.0)
project(VulkanSetup VERSION 0.1.0)

include(CTest)
enable_testing()

if(WIN32)
    message("Using custom FindSDL2.cmake for windows")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/cmake/Win/")
endif()

if(UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    message("Enabling gcc stl debug checks")
    add_compile_options(-D_GLIBCXX_DEBUG)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Enabling vulkan validation layers")
    add_compile_definitions(USEVKVALIDATIONLAYER)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
endif()

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
find_package(netCDF CONFIG REQUIRED)

set(src_files)
add_subdirectory(src)

set(structures_INCLUDE_DIR
    "src"
)
include_directories(${Vulkan_INCLUDE_DIR};${SDL2_INCLUDE_DIRS};${netCDF_INCLUDE_DIR};${structures_INCLUDE_DIR})
add_executable(VulkanSetup ${src_files})
if(WIN32)
set(netCDF_LIBRARIES netCDF::netcdf)
endif()
target_link_libraries(VulkanSetup PRIVATE Vulkan::Vulkan ${SDL2_LIBRARIES} ${netCDF_LIBRARIES})
target_compile_features(VulkanSetup PRIVATE cxx_std_17)
add_subdirectory(shader)

message ("Copying fonts")
file(GLOB FONTS
"${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf"
)
file(COPY ${FONTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fonts)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
