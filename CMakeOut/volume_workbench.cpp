#include "workbenches/volume_workbench.hpp"
#include <util/vk_initializers.hpp>
#include <util/vma_initializers.hpp>
#include <util/vk_util.hpp>
#include <pipelines/volume_renderer.hpp>
#include <util/imgui_util.hpp>
#include <imgui/imgui_stdlib.h>
#include <util/util.hpp>
#include <util/brush_util.hpp>
#include <algorithm>

namespace workbenches {

    volume_workbench::volume_workbench(const std::string_view id) :
        workbench(id)
    {
        _update_plot_image();
    }

    void volume_workbench::_update_plot_image() {
        // waiting for the device to avoid destruction errors
        auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
        if (plot_data.ref_no_track().image)
            util::vk::destroy_image(plot_data.ref_no_track().image);
        if (plot_data.ref_no_track().image_view)
            util::vk::destroy_image_view(plot_data.ref_no_track().image_view);
        if (plot_data.ref_no_track().image_descriptor)
            util::imgui::free_image_descriptor_set(plot_data.ref_no_track().image_descriptor);
        auto image_info = util::vk::initializers::imageCreateInfo(plot_data.read().image_format, { plot_data.read().width, plot_data.read().height, 1 }, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        auto alloc_info = util::vma::initializers::allocationCreateInfo();
        std::tie(plot_data.ref_no_track().image, plot_data.ref_no_track().image_view) = util::vk::create_image_with_view(image_info, alloc_info);
        plot_data.ref_no_track().image_descriptor = util::imgui::create_image_descriptor_set(plot_data.read().image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // updating the image layout
        auto image_barrier = util::vk::initializers::imageMemoryBarrier(plot_data.ref_no_track().image.image, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        util::vk::convert_image_layouts_execute(image_barrier);
    }



    void volume_workbench::show() {
        if (!active)
            return;

        std::string path;
        ImGui::Begin("Add Data");
        ImGui::InputText("Data path", &path);
        if (ImGui::Button("Load and Render")) {
            render_plot();
        }
        ImGui::End();

    }

    void volume_workbench::render_plot()
    {
        if (logger.logging_level >= logging::level::l_5)
            logger << logging::info_prefix << " parallel_coordinates_workbench::render_plot()" << logging::endl;
        pipelines::volume_renderer::render_info render_info{
            *this,  // workbench (is not changed, the renderer only reads information)
            {},     // wait_semaphores;
            {}      // signal_semaphores;
        };
        pipelines::volume_renderer::instance().render(render_info);
    }

}