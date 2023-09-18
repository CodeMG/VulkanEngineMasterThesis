#include <SDL.h> //For SDL
#include <SDL_vulkan.h> //For Vulkan surfaces on SDL windows
#include <iostream> // For cout
#include <structures/vk_context.hpp> // For VKContextInitInfo
#include <util/vk_initializers.hpp> //For all the preprogrammed initializers
#include <imgui/imgui_impl_vulkan.h> //Imgui for vulkan
#include <imgui/imgui_impl_sdl.h>
#include <structures/stager.hpp>
#include <util/imgui_util.hpp>
#include <util/workbenches_util.hpp>
#include <util/global_descriptor_set_util.hpp>
#include <imgui/imgui_internal.h>
#include <structures/workbench_base.hpp>
#include <structures/frame_limiter.hpp>
int main3(int argc, char* argv[]) {
    SDL_Window* window{};
    ImGui_ImplVulkanH_Window    imgui_window_data;
    constexpr int               min_image_count{ 2 };
    const std::string_view      log_window_name{ "log window" };

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) //Initialize SDL
    {
        std::cout << "[error] " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);
    window = SDL_CreateWindow("PCViewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags); //Create SDL Window
    SDL_EventState(SDL_DROPFILE, SDL_ENABLE); //Enable drag and drop

    //Prerequisits for vulkan
    uint32_t instance_extension_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &instance_extension_count, NULL);
    std::vector<const char*> instance_extensions(instance_extension_count);
    SDL_Vulkan_GetInstanceExtensions(window, &instance_extension_count, instance_extensions.data());
    instance_extensions.push_back("VK_KHR_get_physical_device_properties2");
    std::vector<const char*> instance_layers;
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");

    std::vector<const char*> device_extensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME };
    VkPhysicalDeviceVulkan12Features vk_12_features = util::vk::initializers::physicalDeviceVulkan12Features();
    vk_12_features.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 device_features = util::vk::initializers::physicalDeviceFeatures2();
    device_features.pNext = &vk_12_features;

    //Vulkan init
    structures::VkContextInitInfo vk_init{
        -1, //-1 for autodetection
        VK_API_VERSION_1_2,
        "PCViewer",
        instance_layers,
        instance_extensions,
        device_extensions,
        device_features
    };

    auto chosen_gpu = globals::vk_context.init(vk_init); //Initialize Vulkan

    // global stager init
    globals::stager.init(); //Creates the stager buffer (?)

    // imgui init
    if (SDL_Vulkan_CreateSurface(window, globals::vk_context.instance, &imgui_window_data.Surface) == SDL_FALSE) {
        
        return -1;
    }

    int w, h;
    SDL_GetWindowSize(window, &w, &h);

    const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    imgui_window_data.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(globals::vk_context.physical_device, imgui_window_data.Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace); //Surface format

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_IMMEDIATE_KHR };   // current workaround, otherwise on linux lagging
    imgui_window_data.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(globals::vk_context.physical_device, imgui_window_data.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes)); //Present mode
    ImGui_ImplVulkanH_CreateOrResizeWindow(globals::vk_context.instance, globals::vk_context.physical_device, globals::vk_context.device, &imgui_window_data, globals::vk_context.graphics_queue_family_index, globals::vk_context.allocation_callbacks, w, h, min_image_count); 

    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;
    std::vector<float> font_sizes{ 10.f, 15.f, 25.f };
    util::imgui::load_fonts("fonts/", font_sizes);
    ImGui::GetIO().FontDefault = ImGui::GetIO().Fonts->Fonts[1];

    ImGui_ImplSDL2_InitForVulkan(window);

    //Put all the vulkan stuff into imgui
    ImGui_ImplVulkan_InitInfo& init_info = globals::imgui.init_info;
    init_info.Instance = globals::vk_context.instance;
    init_info.PhysicalDevice = globals::vk_context.physical_device;
    init_info.Device = globals::vk_context.device;
    init_info.QueueFamily = globals::vk_context.graphics_queue_family_index;
    init_info.Queue = globals::vk_context.graphics_queue;
    init_info.PipelineCache = {};
    init_info.DescriptorPool = util::imgui::create_desriptor_pool();;
    init_info.Allocator = globals::vk_context.allocation_callbacks;
    init_info.MinImageCount = min_image_count;
    init_info.ImageCount = imgui_window_data.ImageCount;
    init_info.CheckVkResultFn = util::check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, imgui_window_data.RenderPass);

    auto setup_command_pool = util::vk::create_command_pool(util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index));
    auto setup_commands = util::vk::create_begin_command_buffer(setup_command_pool);
    auto setup_fence = util::vk::create_fence(util::vk::initializers::fenceCreateInfo());
    ImGui_ImplVulkan_CreateFontsTexture(setup_commands);
    util::vk::end_commit_command_buffer(setup_commands, globals::vk_context.graphics_queue, {}, {}, {}, setup_fence);
    auto res = vkWaitForFences(globals::vk_context.device, 1, &setup_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    util::vk::destroy_fence(setup_fence);
    util::vk::destroy_command_pool(setup_command_pool);
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    // workbenches setup
    util::workbench::setup_default_workbenches();
    util::global_descriptors::setup_default_descriptors();


    // main loop ---------------------------------------------------------------------
    bool                        done = false;
    bool                        rebuild_swapchain = false;
    int                         swapchain_width = 0, swapchain_height = 0;
    bool                        first_frame = true;
    structures::frame_limiter   frame_limiter;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            else if (event.type == SDL_DROPFILE) {       // In case if dropped file
                //globals::paths_to_open.push_back(std::string(event.drop.file));
                SDL_free(event.drop.file);              // Free dropped_filedir memory;
            }
        }

        if (rebuild_swapchain && swapchain_width > 0 && swapchain_height > 0) {
            ImGui_ImplVulkan_SetMinImageCount(min_image_count);
            ImGui_ImplVulkanH_CreateOrResizeWindow(globals::vk_context.instance, globals::vk_context.physical_device, globals::vk_context.device, &imgui_window_data, globals::vk_context.graphics_queue_family_index, globals::vk_context.allocation_callbacks, swapchain_width, swapchain_height, min_image_count);
            imgui_window_data.FrameIndex = 0;
        }

        // start imgui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        
        // main docking window with menu bar
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGuiWindowFlags dockingWindow_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("MainDockWindow", NULL, dockingWindow_flags);
        ImGui::PopStyleVar(3);
        ImGuiID main_dock_id = ImGui::GetID("MainDock");
        if (first_frame) {
            ImGui::DockBuilderRemoveNode(main_dock_id);
            ImGuiDockNodeFlags dockSpaceFlags = 0;
            dockSpaceFlags |= ImGuiDockNodeFlags_DockSpace;
            ImGui::DockBuilderAddNode(main_dock_id, dockSpaceFlags);
            ImGui::DockBuilderSetNodeSize(main_dock_id, { viewport->WorkSize.x, viewport->WorkSize.y });
            ImGuiID main_dock_bottom, main_dock_top, main_dock_lowest;
            ImGui::DockBuilderSplitNode(main_dock_id, ImGuiDir_Down, .3f, &main_dock_bottom, &main_dock_top);
            ImGui::DockBuilderSplitNode(main_dock_bottom, ImGuiDir_Down, .1f, &main_dock_lowest, &main_dock_bottom);
            ImGui::DockBuilderDockWindow(globals::primary_workbench->id.data(), main_dock_bottom);
            ImGui::DockBuilderDockWindow(globals::secondary_workbench->id.data(), main_dock_top);
            ImGui::DockBuilderDockWindow(log_window_name.data(), main_dock_lowest);
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(main_dock_bottom);
            node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
            node = ImGui::DockBuilderGetNode(main_dock_lowest);
            node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoDocking | ImGuiDockNodeFlags_NoResize;
            ImGui::DockBuilderFinish(main_dock_id);
        }
        auto id = ImGui::DockBuilderGetNode(main_dock_id)->SelectedTabId;
        ImGui::DockSpace(main_dock_id, {}, ImGuiDockNodeFlags_None);

        ImGui::Begin(log_window_name.data());
        ImGui::SetWindowFontScale(.8f);
        for (int i : util::i_range(logger.buffer_size)) {
            auto last_line = logger.get_last_line(logger.buffer_size - 1 - i);
            if (last_line.empty())
                continue;

            if (std::string_view(last_line).substr(0, logging::warning_prefix.size()) == logging::warning_prefix)
                ImGui::TextColored({ .8f, .8f, 0, 1 }, "%s", last_line.c_str());
            else if (std::string_view(last_line).substr(0, logging::error_prefix.size()) == logging::error_prefix)
                ImGui::TextColored({ .8f, 0, .2f, 1 }, "%s", last_line.c_str());
            else if (std::string_view(last_line).substr(0, logging::vulkan_validation_prefix.size()) == logging::vulkan_validation_prefix)
                ImGui::TextColored({ .8f, .8f, .8f, 1 }, "%s", last_line.c_str());
            else
                ImGui::Text("%s", last_line.c_str());
        }
        ImGui::SetScrollHereY(1);
        ImGui::End();   // log window

        ImGui::End();   // Main Window
        

        /// ////////////////////////////////////////////////////////////////////////////////////////


        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool minimized = draw_data->DisplaySize.x <= 0 || draw_data->DisplaySize.y <= 0;
        {   // rendering scope
            std::scoped_lock lock(*globals::vk_context.graphics_mutex);
            if (!minimized)
                util::imgui::frame_render(&imgui_window_data, draw_data);

            
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            

            if (!minimized)
                std::tie(rebuild_swapchain, swapchain_width, swapchain_height) = util::imgui::frame_present(&imgui_window_data, window);
        }
        res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);


        frame_limiter.end_frame();
        first_frame = false;
    }

    res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    ImGui_ImplVulkanH_DestroyWindow(globals::vk_context.instance, globals::vk_context.device, &imgui_window_data, globals::vk_context.allocation_callbacks);
    globals::stager.cleanup();
    globals::vk_context.cleanup();

    SDL_DestroyWindow(window);
    SDL_Quit();

}