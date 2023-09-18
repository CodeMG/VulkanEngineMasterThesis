#include "pipelines/volume_renderer.hpp"
#include <util/vk_initializers.hpp>
#include <util/vma_initializers.hpp>
#include <util/vk_util.hpp>
#include <util/vma_util.hpp>
#include <util/file_util.hpp>
#include <array>
#include <workbenches/volume_workbench.hpp>
#include <structures/array_struct.hpp>
#include <util/global_descriptor_set_util.hpp>

namespace pipelines
{
    volume_renderer::volume_renderer()
    {
        auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index);
        _command_pool = util::vk::create_command_pool(pool_info);
        auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
        _render_fence = util::vk::create_fence(fence_info);
    }

    void volume_renderer::_pre_render_commands(VkCommandBuffer commands, const output_specs& output_specs)
    {
        const auto& pipe_data = _pipelines[output_specs];
        auto begin_info = util::vk::initializers::renderPassBeginInfo(pipe_data.render_pass, pipe_data.framebuffer, { 0, 0, output_specs.width, output_specs.height }, VkClearValue{});
        vkCmdBeginRenderPass(commands, &begin_info, {});
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_data.pipeline);
        VkDescriptorSet sets[3] = { globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->descriptor_set, globals::descriptor_sets["data"]->descriptor_set, globals::descriptor_sets["tf"]->descriptor_set };
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_data.pipeline_layout, 0, 3, sets, 0, {});
        VkViewport viewport{};
        viewport.width = static_cast<float>(output_specs.width);
        viewport.height = static_cast<float>(output_specs.height);
        viewport.maxDepth = 1;
        vkCmdSetViewport(commands, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = { output_specs.width, output_specs.height };
        vkCmdSetScissor(commands, 0, 1, &scissor);
    }

    void volume_renderer::_post_render_commands(VkCommandBuffer commands, const output_specs& output_specs, VkFence fence, util::memory_view<VkSemaphore> wait_semaphores, util::memory_view<VkSemaphore> signal_semaphores)
    {
        vkCmdEndRenderPass(commands);
        std::vector<VkPipelineStageFlags> stage_flags(wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
        std::scoped_lock lock(*globals::vk_context.graphics_mutex);
        util::vk::end_commit_command_buffer(commands, globals::vk_context.graphics_queue, wait_semaphores, stage_flags, signal_semaphores, fence);
    }

    const volume_renderer::pipeline_data& volume_renderer::get_or_create_pipeline(const output_specs& output_specs) {
        if (!_pipelines.contains(output_specs)) {
            if (logger.logging_level >= logging::level::l_4) {
                std::stringstream ss; ss << util::memory_view<const uint32_t>(util::memory_view(output_specs));
                logger << "[info] volume_renderer::get_or_create_pipeline() creating new pipeline for output_specs " << ss.str() << logging::endl;
            }

            if (_pipelines.size() > max_pipeline_count) {
                auto [pipeline, time] = *std::min_element(_pipeline_last_use.begin(), _pipeline_last_use.end(), [](const auto& l, const auto& r) {return l.second < r.second; });
                auto p = pipeline;
                auto [key, val] = *std::find_if(_pipelines.begin(), _pipelines.end(), [&](const auto& e) {return e.second.pipeline == p; });
                util::vk::destroy_pipeline(val.pipeline);
                util::vk::destroy_pipeline_layout(val.pipeline_layout);
                util::vk::destroy_framebuffer(val.framebuffer);
                util::vk::destroy_render_pass(val.render_pass);
                if (val.multi_sample_image)
                    util::vk::destroy_image(val.multi_sample_image);
                if (val.multi_sample_view)
                    util::vk::destroy_image_view(val.multi_sample_view);
                _pipeline_last_use.erase(pipeline);
                _pipelines.erase(key);
            }

            pipeline_data& pipe_data = _pipelines[output_specs];
            // creating the rendering buffers  -------------------------------------------------------------------------------
            // output image after multisample reduction (already given by parallel coordinates workbench)

            // multisample image
            if (output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT) {
                auto image_info = util::vk::initializers::imageCreateInfo(output_specs.format, { output_specs.width, output_specs.height, 1 }, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TYPE_2D, 1, 1, output_specs.sample_count);
                auto allocation_info = util::vma::initializers::allocationCreateInfo();
                std::tie(pipe_data.multi_sample_image, pipe_data.multi_sample_view) = util::vk::create_image_with_view(image_info, allocation_info);

                // updating the image layout
                auto image_barrier = util::vk::initializers::imageMemoryBarrier(pipe_data.multi_sample_image.image, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, {}, {}, {}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                util::vk::convert_image_layouts_execute(image_barrier);
            }

            // render pass
            std::vector<VkAttachmentDescription> attachments;
            VkAttachmentDescription attachment = util::vk::initializers::attachmentDescription(output_specs.format);
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(attachment);
            if (output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT) {
                attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                attachment.samples = output_specs.sample_count;
                attachments.push_back(attachment);
            }
            std::vector<VkAttachmentReference> attachment_references;
            VkAttachmentReference attachment_reference{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            attachment_references.push_back(attachment_reference);
            util::memory_view<VkAttachmentReference> resolve_reference{};
            if (output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT) {
                attachment_references.back().attachment = 1;    // normal rendering goes to attachment 1 in multisampling case (at index 1 multisample image is attached)
                attachment_reference.attachment = 0;
                resolve_reference = util::memory_view(attachment_reference);
            }
            auto subpass_description = util::vk::initializers::subpassDescription(VK_PIPELINE_BIND_POINT_GRAPHICS, {}, attachment_references, resolve_reference);

            auto render_pass_info = util::vk::initializers::renderPassCreateInfo(attachments, subpass_description);
            pipe_data.render_pass = util::vk::create_render_pass(render_pass_info);

            // framebuffer
            std::vector<VkImageView> image_views{ output_specs.plot_image_view };
            if (output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT)
                image_views.push_back(pipe_data.multi_sample_view);
            auto framebuffer_info = util::vk::initializers::framebufferCreateInfo(pipe_data.render_pass, image_views, output_specs.width, output_specs.height, 1);
            pipe_data.framebuffer = util::vk::create_framebuffer(framebuffer_info);

            // creating the rendering pipeline -------------------------------------------------------------------------------

            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants), 0);
            assert(globals::descriptor_sets.contains(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            std::vector<VkDescriptorSetLayout> layouts_vector;
            layouts_vector.push_back(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout);
            layouts_vector.push_back(globals::descriptor_sets["data"]->layout);
            layouts_vector.push_back(globals::descriptor_sets["tf"]->layout);
            
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(util::memory_view(layouts_vector), util::memory_view(push_constant_range)); //GEÄNDERT
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            // pipeline creation
            auto pipeline_rasterizer = util::vk::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_CLOCKWISE); //VK_FRONT_FACE_CLOCKWISE;VK_POLYGON_MODE_FILL;VK_CULL_MODE_NONE
            pipeline_rasterizer.lineWidth = 1;

            auto pipeline_color_blend_attachment = util::vk::initializers:: pipelineColorBlendAttachmentStateStandardAlphaBlend();

            auto pipeline_color_blend = util::vk::initializers::pipelineColorBlendStateCreateInfo(pipeline_color_blend_attachment);

            auto pipeline_viewport = util::vk::initializers::pipelineViewportStateCreateInfo(1, 1);

            std::vector<VkDynamicState> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            auto pipeline_dynamic_states = util::vk::initializers::pipelineDynamicStateCreateInfo(dynamic_states);

            auto pipeline_depth_stencil = util::vk::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

            auto pipeline_multi_sample = util::vk::initializers::pipelineMultisampleStateCreateInfo(output_specs.sample_count);

            auto pipeline_vertex_state = util::vk::initializers::pipelineVertexInputStateCreateInfo();//(vertex_input_binding, vertex_input_attribute);

            VkShaderModule vertex_module = util::vk::create_shader_module(vertex_shader_path);
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path);

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module);

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, pipe_data.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipline(pipeline_create_info);

            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);

        }
        _pipeline_last_use[_pipelines[output_specs].pipeline] = std::chrono::system_clock::now();
        return _pipelines[output_specs];
    }

    const structures::buffer_info& volume_renderer::get_or_resize_info_buffer(size_t byte_size) {
        if (byte_size > _attribute_info_buffer.size) {
            if (_attribute_info_buffer)
                util::vk::destroy_buffer(_attribute_info_buffer);

            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, byte_size);
            auto allocation_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            _attribute_info_buffer = util::vk::create_buffer(buffer_info, allocation_info);
        }
        return _attribute_info_buffer;
    }

    volume_renderer& volume_renderer::instance() {
        static volume_renderer renderer;
        return renderer;
    }

    void volume_renderer::render(const render_info& info) {
        auto data_type = structures::volume_renderer::data_type::float_t;
        output_specs out_specs{
            info.workbench.plot_data.read().image_view,
            info.workbench.plot_data.read().image_format,
            info.workbench.plot_data.read().image_samples,
            info.workbench.plot_data.read().width,
            info.workbench.plot_data.read().height,
            info.workbench.render_type.read(),
            data_type,
        };
        auto pipeline_info = get_or_create_pipeline(out_specs);

        auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);  // wait indefenitely for prev rendering
        vkResetFences(globals::vk_context.device, 1, &_render_fence);

        if (_render_commands.size())
            vkFreeCommandBuffers(globals::vk_context.device, _command_pool, static_cast<uint32_t>(_render_commands.size()), _render_commands.data());
        _render_commands.resize(1);
        _render_commands[0] = util::vk::create_begin_command_buffer(_command_pool);
        _pre_render_commands(_render_commands[0], out_specs);
        VkClearAttachment clear_value{};
        clear_value.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clear_value.clearValue.color.float32[0] = 0;
        clear_value.clearValue.color.float32[1] = 0;
        clear_value.clearValue.color.float32[2] = 0;
        clear_value.clearValue.color.float32[3] = 1;
        VkClearRect clear_rect{};
        clear_rect.layerCount = 1;
        clear_rect.rect.extent = { out_specs.width, out_specs.height };
        vkCmdClearAttachments(_render_commands[0], 1, &clear_value, 1, &clear_rect);
        push_constants pc{};
        pc.aspect_ratio = 2000 / 480;
        pc.focal_length = 3.14f/4;
        pc.f = 10.0;
        pc.n = 0.1;
        //camera
        pc.cameraXPos = info.workbench.cameraXPos;
        pc.cameraYPos = info.workbench.cameraYPos;
        pc.cameraZPos = info.workbench.cameraZPos;
        pc.cameraXAngle = info.workbench.cameraXAngle;
        pc.cameraYAngle = info.workbench.cameraYAngle;
        pc.cameraZAngle = info.workbench.cameraZAngle;
        pc.frame = info.workbench.frame;
        vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(_render_commands.back(), 36, static_cast<uint32_t>(36), 0, static_cast<uint32_t>(0));
        // committing last command buffer
        _post_render_commands(_render_commands.back(), out_specs, _render_fence);
    }
}