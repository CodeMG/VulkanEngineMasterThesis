#include "Correlator.hpp"

#include <util/vk_util.hpp>
#include <util/vma_util.hpp>
#include <util/vk_initializers.hpp>
#include <util/file_util.hpp>
#include <structures/drawlists.hpp>

namespace pipelines
{
    correlator::correlator()
    {
        auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
        _command_pool = util::vk::create_command_pool(pool_info);
        auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
        _correlation_fence = util::vk::create_fence(fence_info);

        // pipeline creation
        auto shader_module = util::vk::create_shader_module(compute_shader_path);
        auto stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, shader_module);

        VkPushConstantRange push_constant{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants) };
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _correlation_pipeline_layout = util::vk::create_pipeline_layout(layout_info);

        auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(_correlation_pipeline_layout, stage_create_info);
        _correlation_pipeline = util::vk::create_compute_pipline(pipeline_info);

        vkDestroyShaderModule(globals::vk_context.device, shader_module, globals::vk_context.allocation_callbacks);
    }

    correlator& correlator::instance()
    {
        static correlator singleton;
        return singleton;
    }

    void correlator::correlate(const correlation_info& info)
    {

        push_constants pc{};
        pc.data_size = static_cast<uint32_t>(0);
        auto res = vkWaitForFences(globals::vk_context.device, 1, &_correlation_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
        res = vkResetFences(globals::vk_context.device, 1, &_correlation_fence); util::check_vk_result(res);
        if (_command_buffer)
            vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
        _command_buffer = util::vk::create_begin_command_buffer(_command_pool);
        vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _correlation_pipeline);
        vkCmdPushConstants(_command_buffer, _correlation_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t dispatch_x = ((pc.data_size + 31) / 32 + shader_local_size - 1) / shader_local_size;
        vkCmdDispatch(_command_buffer, dispatch_x, 1, 1);
        util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue, info.wait_semaphores, info.wait_flags, info.signal_semaphores, _correlation_fence);
    }
}