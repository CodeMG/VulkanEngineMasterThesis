file(GLOB main_src
    "main.cpp"
)

file(GLOB imgui_src
    "imgui/*.cpp"
    "imgui/*.h"
    "imgui/*.hpp"
    "imgui_file_dialog/*.cpp"
    "imgui_file_dialog/*.h"
    "imgui_file_dialog/*.hpp"
    "imgui_nodes/*.cpp"
    "imgui_nodes/*.h"
    "imgui_nodes/*.hpp"
)

file(GLOB pipeline_src
    "pipelines/*.cpp"
    "pipelines/*.h"
    "pipelines/*.hpp"
)

file(GLOB globals_src
    "structures/*.cpp"
    "structures/*.h"
    "structures/*.hpp"
)

file(GLOB util_src
    "util/*.cpp"
    "util/*.h"
    "util/*.hpp"
)

file(GLOB vk_src
    "vulkan/*.cpp"
    "vulkan/*.h"
    "vulkan/*.hpp"
)

file(GLOB workbenches_src
    "workbenches/*.cpp"
    "workbenches/*.h"
    "workbenches/*.hpp"
)

set(src_files "${src_files};${main_src};${imgui_src};${pipeline_src};${globals_src};${util_src};${vk_src};${workbenches_src}" PARENT_SCOPE)