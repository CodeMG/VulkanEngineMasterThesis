#pragma once

#include <structures/brushes.hpp>
#include <string_view>
#include <structures/buffer_info.hpp>
#include <structures/enum_names.hpp>
#include <util/memory_view.hpp>

namespace pipelines {
    class correlator {
        struct push_constants {
            uint32_t        data_size;
        };

        const std::string_view  compute_shader_path{ "shader/correlation.comp.spv" };
        const uint32_t          shader_local_size{ 256 };

        VkPipeline          _correlation_pipeline{};
        VkPipelineLayout    _correlation_pipeline_layout{};
        VkFence             _correlation_fence{};
        VkCommandPool       _command_pool{};
        VkCommandBuffer     _command_buffer{};

        correlator();

    public:

        struct correlation_info {
            std::string_view                drawlist_id{};
            util::memory_view<VkSemaphore>  wait_semaphores{};
            util::memory_view<VkSemaphoreWaitFlags> wait_flags{};
            util::memory_view<VkSemaphore>  signal_semaphores{};
        };

        correlator(const correlator&) = delete;
        correlator& operator=(const correlator&) = delete;

        static correlator& instance();

        void correlate(const correlation_info& info);
    };
}