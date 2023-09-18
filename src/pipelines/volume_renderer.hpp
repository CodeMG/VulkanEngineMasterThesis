#pragma once
#include <util/memory_view.hpp>
#include <structures/vk_context.hpp>
#include <robin_hood_map/robin_hood.h>
#include <structures/change_tracker.hpp>
#include <structures/rendering_structs.hpp>
#include <structures/drawlists.hpp>
#include <optional>
#include <chrono>
#include <imgui/imgui.h>

namespace workbenches {
    class volume_workbench;
}

namespace pipelines {

    class volume_renderer {
        using output_specs = structures::volume_renderer::output_specs;
        using pipeline_data = structures::volume_renderer::pipeline_data;
        using time_point = std::chrono::time_point<std::chrono::system_clock>;

        struct push_constants {
            float focal_length;
            float aspect_ratio;
            float n;
            float f;
            //camerastuff
            float cameraXPos;
            float cameraYPos;
            float cameraZPos;
            float cameraXAngle;
            float cameraYAngle;
            float cameraZAngle;
            int frame;
        };

        const std::string_view vertex_shader_path{ "shader/volume_renderer.vert.spv" };
        const std::string_view large_vis_vertex_shader_path{ "" };
        const std::string_view fragment_shader_path{ "shader/volume_renderer.frag.spv" };
        const std::string_view compute_shader_path{"shader/correlation.comp.spv"};

        // vulkan resources that are the same for all drawlists/parallel_coordinates_windows
        structures::buffer_info                                 _attribute_info_buffer{};
        VkCommandPool                                           _command_pool{};
        VkFence                                                 _render_fence{};    // needed as only a single attribute info buffer exists
        std::vector<VkCommandBuffer>                            _render_commands{};
        

        robin_hood::unordered_map<output_specs, pipeline_data>  _pipelines{};
        robin_hood::unordered_map<VkPipeline, time_point>       _pipeline_last_use{};

        const pipeline_data& get_or_create_pipeline(const output_specs& output_specs);
        const structures::buffer_info& get_or_resize_info_buffer(size_t byte_size); 

        volume_renderer();

        void _pre_render_commands(VkCommandBuffer commands, const output_specs& output_specs);
        void _post_render_commands(VkCommandBuffer commands, const output_specs& output_specs, VkFence fence = {}, util::memory_view<VkSemaphore> wait_semaphores = {}, util::memory_view<VkSemaphore> signal_semaphores = {});

    public:

        VkBuffer                                                _vertex_buffer;
        VkBuffer                                                _index_buffer;

        struct render_info {
            const workbenches::volume_workbench& workbench;
            util::memory_view<VkSemaphore>                      wait_semaphores;
            util::memory_view<VkSemaphore>                      signal_semaphores;
        };

        volume_renderer(const volume_renderer&) = delete;
        volume_renderer& operator=(const volume_renderer&) = delete;

        static volume_renderer& instance();

        void render(const render_info& info);

        uint32_t max_pipeline_count{ 20 };
    };

};
