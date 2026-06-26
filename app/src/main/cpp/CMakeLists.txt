cmake_minimum_required(VERSION 3.22.1)
project(vkbench)

add_library(vkbench SHARED vk_bench.cpp)

find_library(vulkan-lib vulkan)
find_library(log-lib log)
target_link_libraries(vkbench ${vulkan-lib} ${log-lib})
