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
#include <structures/stager.hpp>
#include <fstream>
#include <structures/vk_context.hpp>
#include "util/global_descriptor_set_util.hpp"
#include <math.h>
#include <util/dataset_util.hpp>

#include <chrono>

#include <iostream>
#include <string>
namespace workbenches {

    //Transferfunction stuff
    //Graphical Side
    struct Dot
    {
        ImVec2 pos;
        ImVec2 posDelta;
        bool isDragging;
        bool isSelected;
        bool isXLocked;
        bool isYLocked;
        float color[3];
        float colorDelta[3];
        Dot(ImVec2 pos) : pos(pos), isDragging(false),isSelected(false), isXLocked(false) ,isYLocked(false){}
    };
    std::vector<Dot> histogramDots;
    std::vector<Dot> gradientDots;
    //Data side
    const int binCount = 8;
    float bins[5][binCount];
    float my_color[binCount][4];
    int selectedBin = 0;
    //structures::buffer_info tfBuffer;
    structures::image_info tfImage;
    struct TF {
        float my_color[binCount][4];
    } tf;

    //Data
    util::dataset::open_internals::load_result<float> res;
    std::vector<uint32_t> dims;
    std::vector<float> pearsonData;
    std::vector<float> spearmenData;
    std::vector<float> kendallData;
    std::vector<float> ksgData;

    structures::image_info destination;

    //KorrelationAlgorithms
    int referencePoint[4] = {176,125,10,1};
    int selectedAlgo = 0;

    //Optimization stuff
    std::vector<double> referenceImage;
    float error = -1.0f;
    std::vector<std::vector<float>> tf_delta;
    bool optimize = false;

    //Misc
    int frameCounter = 0;

    void moveData();
    void pearson_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level, std::vector<float>* out);
    void spearmen_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level,std::vector<float>* out);
    void kendall_rank_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level,std::vector<float>* out);
    void KSG_mutual_information(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level, std::vector<float>* out);
    volume_workbench::volume_workbench(const std::string_view id) :
        workbench(id)
    {
        for (int i = 0; i < binCount; i++) {
            bins[0][i] = 0.0;
            bins[1][i] = 0.0;
            bins[2][i] = 0.0;
            bins[3][i] = 0.0;
        }

        tf_delta = std::vector<std::vector<float>>(binCount);
        for (int i = 0; i < binCount;i++) {
            tf_delta[i] = std::vector<float>(3);
        }


        res = util::dataset::open_internals::open_netcdf<float>("D:/MasterArbeit/Data/TestData/necker_t5_e100_tk.nc");
        uint32_t ref_level = 10;
        uint32_t ref_lat = 176;
        uint32_t ref_lon = 125;
        uint32_t ref_member = 0;
        //ref_level = 10;
        //ref_lat = 50;
        //ref_lon = 50;
        float d = res.data({ ref_member, ref_level, ref_lat, ref_lon }, 0);
        dims = res.data.dimension_sizes;
        //pearson_correlation(&res,ref_lat,ref_lon,ref_level,&pearsonData); 
        //spearmen_correlation(&res, ref_lat, ref_lon, ref_level,&data); 
        //kendall_rank_correlation(&res, ref_lat, ref_lon, ref_level,&kendallData);
        //KSG_mutual_information(&res,ref_lat,ref_lon,ref_level,&data);
        //std::cout << "Correlation: " << corr << std::endl;

        _update_plot_image();
        moveData();
        cameraXPos = 0;
        cameraYPos = 0;
        cameraZPos = -4;
        cameraXAngle = 0;
        cameraYAngle = 0.0f;
        cameraZAngle = 0;
        frame = 0;
       
        //Create the leftmost and rightmost dots for the TF picker:
        //Dot left = {ImVec2(0,)}
        //dots.push_back());
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
        auto image_info = util::vk::initializers::imageCreateInfo(plot_data.read().image_format, { plot_data.read().width, plot_data.read().height, 1 }, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        auto alloc_info = util::vma::initializers::allocationCreateInfo();
        std::tie(plot_data.ref_no_track().image, plot_data.ref_no_track().image_view) = util::vk::create_image_with_view(image_info, alloc_info);
        plot_data.ref_no_track().image_descriptor = util::imgui::create_image_descriptor_set(plot_data.read().image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // updating the image layout
        auto image_barrier = util::vk::initializers::imageMemoryBarrier(plot_data.ref_no_track().image.image, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        util::vk::convert_image_layouts_execute(image_barrier);
    }

    
    bool pairSortKeyFunction(std::pair<float, uint32_t> i, std::pair<float, uint32_t> j) { return (i.first < j.first); }
    bool dotSortFunction(Dot i, Dot j) { return (i.pos.x < j.pos.x); }

    //inspired by https://en.wikipedia.org/wiki/Digamma_function#Asymptotic_expansion
    //Wikipedia cites: https://www.uv.es/~bernardo/1976AppStatist.pdf
    double digamma(double x) {
        if (x < 8.5) {
            return digamma(x + 1.0) - (1.0 / x);
        }
        else {
            return log(x) - (1.0 / (2.0 * x)) - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x * x * x * x) - 1.0 / (252.0 * pow(x, 6)) + 1.0 / (240.0 * pow(x, 8)) - 5 / (660 * pow(x, 10)) + 691 / (32760 * pow(x, 12)) - 1 / (12 * pow(x, 14));
        }
    }
    void KSG_mutual_information(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level, std::vector<float>* out){
        std::vector<uint32_t> get(4);
        std::vector<uint32_t> dim = data->data.dimension_sizes;
        int k = 3; //For k-nearest neighbours
        for (uint32_t x = 0; x < dim[1]; x++) { //level
            std::cout << "level: " << x << std::endl;
            for (uint32_t y = 0; y < dim[2]; y++) { //lat
                for (uint32_t z = 0; z < dim[3]; z++) { //lon
                    uint32_t P = 0, Q = 0, X = 0, Y = 0;
                    std::vector<std::pair<float, float>> points;
                    //Create the points
                    for (uint32_t i = 0; i < dim[0]; i++) {
                        get[0] = i;
                        get[1] = reference_level;
                        get[2] = reference_lat;
                        get[3] = reference_lon;
                        float ref = data->data(get, 0);
                        get[0] = i;
                        get[1] = x;
                        get[2] = y;
                        get[3] = z;
                        float voxel = data->data(get, 0);
                        if (!isnan(ref) && !isnan(voxel)) {
                            points.push_back(std::pair<float, float>(ref, voxel));
                        }
                    }
                    if (points.size() < k+1) {
                        out->push_back(0);
                        continue;
                    }
                    //Calculate biggest distance in k distances
                    std::vector<float> biggest_x_diffs;
                    std::vector<float> biggest_y_diffs;
                    for each (auto point in points) {
                        std::vector<std::pair<float,uint32_t>> distances;
                        int counter = 0;
                        for each (auto other_point in points) { //calculate all the distances from point to other_point
                            distances.push_back(std::pair<float,int>(sqrt((other_point.first-point.first)*(other_point.first - point.first)+(other_point.second - point.second)*(other_point.second - point.second)),counter));
                            counter++;
                        }
                        //Sort
                        std::sort(distances.begin(), distances.end(), pairSortKeyFunction); // sort them to find the k-nearest neighbours

                        //Find the biggest Xdiff and the biggest Ydiff in the k nearest neighbours
                        float xDiff = 0;
                        float yDiff = 0;
                        for (int i = 0; i < k+1;i++) {
                            int id = distances[i].second;
                            float current_x_diff = abs(points[id].first - point.first);
                            float current_y_diff = abs(points[id].second - point.second);
                            xDiff = fmax(xDiff,current_x_diff);
                            yDiff = fmax(yDiff,current_y_diff);
                        }
                        biggest_x_diffs.push_back(xDiff);
                        biggest_y_diffs.push_back(yDiff);
                    }

                    //Calculate a and b by getting all the points where the x distance < biggest_x_diffs or y distance < biggest_y_diffs
                    double a = 0;
                    double b = 0;
                    int current_point_index = 0;
                    for each (auto point in points) {
                        int amount_of_x_points_in_radius = 0;
                        int amount_of_y_points_in_radius = 0;
                        for each (auto other_point in points) {
                            float current_x_dist = abs(point.first - other_point.first);
                            float current_y_dist = abs(point.second - other_point.second);
                            if (current_x_dist < biggest_x_diffs[current_point_index]) {
                                amount_of_x_points_in_radius++;
                            }
                            if (current_y_dist < biggest_y_diffs[current_point_index]) {
                                amount_of_y_points_in_radius++;
                            }
                        }
                        current_point_index++;
                        a += digamma(amount_of_x_points_in_radius);
                        b += digamma(amount_of_y_points_in_radius);
                    }
                    a /= points.size();
                    b /= points.size();
                    double c = digamma(k) - (1.0 / k);
                    double d = digamma(points.size());
                    double mi = -a - b + c + d;
                    
                    out->push_back(fmax(mi,0.0001));
                }
            }
        }
    }
    void kendall_rank_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level,std::vector<float>* out) {
        std::vector<uint32_t> get(4);
        std::vector<uint32_t> dim = data->data.dimension_sizes;
        for (uint32_t x = 0; x < dim[1]; x++) { //level
            std::cout << "level: " << x << std::endl;
            for (uint32_t y = 0; y < dim[2]; y++) { //lat
                for (uint32_t z = 0; z < dim[3]; z++) { //lon
                    int P = 0, Q = 0, X = 0, Y = 0;
                    std::vector<std::pair<float, float>> points;
                    //Create the points
                    for (uint32_t i = 0; i < dim[0]; i++) {
                        get[0] = i;
                        get[1] = reference_level;
                        get[2] = reference_lat;
                        get[3] = reference_lon;
                        float ref = data->data(get, 0);
                        //std::cout << "ref: " << ref << std::endl;
                        get[0] = i;
                        get[1] = x;
                        get[2] = y;
                        get[3] = z;
                        float voxel = data->data(get, 0);
                        if (!isnan(ref) && !isnan(voxel)) {
                            points.push_back(std::pair<float,float>(ref,voxel));
                        }
                    }

                    for (int i = 0; i < points.size();i++) {
                        float refX = points[i].first;
                        float refY = points[i].second;
                        for (int j = i+1; j < points.size();j++) {
                            float otherX = points[j].first;
                            float otherY = points[j].second;
                            if (refX > otherX) {
                                if (refY > otherY) {
                                    P++;
                                }
                                else if (refY < otherY) {
                                    Q++;
                                }
                                else {
                                    Y++;
                                }
                            }
                            else if (refX < otherX) {
                                if (refY > otherY) {
                                    Q++;
                                }
                                else if (refY < otherY) {
                                    P++;
                                }
                                else {
                                    Y++;
                                }
                            }
                            else {
                                X++;
                                if (refY == otherY) {
                                    Y++;
                                }
                            }
                        }
                    }
                    //float corr = (float)(P - Q) / sqrt(   (     (float)(P + Q + X)      ) * (   (float)(P + Q + Y)    )  );
                    int n = points.size();
                    float top = (P - Q) * 1.0f;
                    float bottom1 = sqrt(((1.0f / 2.0f) * n * (n - 1)) - X);
                    float bottom2 = sqrt(((1.0f / 2.0f) * n * (n - 1)) - Y);
                    float corr = top / (bottom1 * bottom2);
                    //float corr = (float)(P - Q) / (sqrt(((1.0f/2.0f) * n * (n-1)) - X)* sqrt(((1.0f/2.0f)*n*(n-1))-Y));
                    if(n == 0) {
                      out->push_back(0.5f);
                    }
                    else {
                      out->push_back((corr + 1.0) / 2.0);
                    }
                    
                }
            }
        }
    }

    
    //bool pairSortValueFunction(std::pair<uint32_t,float> i, std::pair<uint32_t, float> j) { return (i.second < j.second); }

    void spearmen_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level, std::vector<float>* out) {
        std::vector<uint32_t> get(4);
        std::vector<uint32_t> dim = data->data.dimension_sizes;
        for (uint32_t x = 0; x < dim[1]; x++) { //level
            std::cout << "level: " << x << std::endl;
            for (uint32_t y = 0; y < dim[2]; y++) { //lat
                for (uint32_t z = 0; z < dim[3]; z++) { //lon
                    double E_x = 0;
                    int E_x_counter = 0;
                    double E_y = 0;
                    int E_y_counter = 0;
                    double E_xy = 0;
                    int E_xy_counter = 0;
                    double E_xx = 0;
                    int E_xx_counter = 0;
                    double E_yy = 0;
                    int E_yy_counter = 0;
                    //First sort for rank
                    std::vector<std::pair<float, uint32_t>> X_rank;
                    std::vector<std::pair<float, uint32_t>> Y_rank;
                    std::vector<std::pair<float, float>> final_X_ranks;
                    std::vector<std::pair<float, float>> final_Y_ranks;
                    for (uint32_t i = 0; i < dim[0]; i++) {
                        get[0] = i;
                        get[1] = reference_level;
                        get[2] = reference_lat;
                        get[3] = reference_lon;
                        float ref = data->data(get, 0);
                        get[0] = i;
                        get[1] = x;
                        get[2] = y;
                        get[3] = z;
                        float voxel = data->data(get, 0);
                        if (!isnan(ref) && !isnan(voxel)) {
                            X_rank.push_back(std::pair<float, uint32_t>(ref,i));
                            Y_rank.push_back(std::pair<float, uint32_t>(voxel, i));
                        }
                        
                    }
                    //If the nan's take over
                    if (X_rank.size() == 0) {
                        out->push_back(0.5f);
                        continue;
                    }
                    std::sort(X_rank.begin(), X_rank.end(),pairSortKeyFunction);
                    std::sort(Y_rank.begin(), Y_rank.end(), pairSortKeyFunction);

                    for (uint32_t i = 0; i < X_rank.size(); i++) {
                        final_X_ranks.push_back(std::pair<float, float>(X_rank[i].first,i));
                        final_Y_ranks.push_back(std::pair<float, float>(Y_rank[i].first,i));
                    }

                    //Create and adjust duplicate ranks
                    float lastValue = 0;
                    double sum = 0;
                    int count = 0;
                    //For X
                    for (int i = 0; i < X_rank.size();i++) {
                        if (i == 0) {
                            lastValue = X_rank[i].first;
                            sum += i;
                            count++;
                            continue;
                        }
                        float currentValue = X_rank[i].first;
                        if (lastValue != currentValue) {
                            double newRank = sum / count;
                            //Adjust the previous ranks
                            for (int j = 1; j <= count;j++) {
                                final_X_ranks[i - j].second = sum;
                            }
                            //Reset
                            sum = 0;
                            count = 0;
                        }
                        else {
                            //If last one, and the last one is equal to the previous one
                            if (i == X_rank.size() -1) {
                                sum += i;
                                count++;
                                double newRank = sum / count;
                                //Adjust the previous ranks
                                for (int j = 1; j <= count; j++) {
                                    final_X_ranks[i - j].second = sum;
                                }

                            }
                        }
                        sum += i;
                        count++;
                    }

                    //For Y
                    lastValue = 0;
                    sum = 0;
                    count = 0;

                    for (int i = 0; i < X_rank.size(); i++) {
                        if (i == 0) {
                            lastValue = Y_rank[i].first;
                            sum += i;
                            count++;
                            continue;
                        }
                        float currentValue = Y_rank[i].first;
                        if (lastValue != currentValue) {
                            double newRank = sum / count;
                            //Adjust the previous ranks
                            for (int j = 1; j <= count; j++) {
                                final_Y_ranks[i - j].second = sum;
                            }
                            //Reset
                            sum = 0;
                            count = 0;
                        }
                        else {
                            //If last one, and the last one is equal to the previous one
                            if (i == X_rank.size() - 1) {
                                sum += i;
                                count++;
                                double newRank = sum / count;
                                //Adjust the previous ranks
                                for (int j = 1; j <= count; j++) {
                                    final_Y_ranks[i - j].second = sum;
                                }

                            }
                        }
                        sum += i;
                        count++;
                    }

                    //Calculate pearson corr
                    for (int i = 0; i < X_rank.size();i++) {
                        double currentXRank = final_X_ranks[i].second;
                        uint32_t y_counter = 0;
                        for (auto it2 = Y_rank.begin(); it2 != Y_rank.end(); it2++) {
                            if (it2->second == X_rank[i].second) {
                                break;
                            }
                            y_counter++;
                        }
                        double currentYRank = final_Y_ranks[y_counter].second;
                        E_x += currentXRank;
                        E_x_counter++;
                        E_xx += currentXRank * currentXRank;
                        E_xx_counter++;

                        E_y += currentYRank;
                        E_y_counter++;
                        E_xy += currentXRank * currentYRank;
                        E_xy_counter++;
                        E_yy += currentYRank * currentYRank;
                        E_yy_counter++;

                        
                    }
                    E_x /= E_x_counter;
                    E_y /= E_y_counter;
                    E_xy /= E_xy_counter;
                    E_xx /= E_xx_counter;
                    E_yy /= E_yy_counter;
                    float corr = (E_xy - (E_x * E_y)) / (sqrt(E_xx - (E_x * E_x)) * sqrt(E_yy - (E_y * E_y)));
                    out->push_back((corr + 1.0) /2.0);
                    
                }
            }
        }
    }

    void pearson_correlation(util::dataset::open_internals::load_result<float>* data, uint32_t reference_lat, uint32_t reference_lon, uint32_t reference_level,std::vector<float>* out) {
        std::vector<uint32_t> dim = data->data.dimension_sizes;
        std::vector<uint32_t> get(4);
        for (uint32_t x = 0; x < dim[1]; x++) { //level
            std::cout << "level: " << x << std::endl;
            for (uint32_t y = 0; y < dim[2]; y++) { //lat
                for (uint32_t z = 0; z < dim[3]; z++) { //lon
                    float E_x = 0;
                    int E_x_counter = 0;
                    float E_y = 0;
                    int E_y_counter = 0;
                    float E_xy = 0;
                    int E_xy_counter = 0;
                    float E_xx = 0;
                    int E_xx_counter = 0;
                    float E_yy = 0;
                    int E_yy_counter = 0;
                    for (uint32_t i = 0; i < dim[0];i++) {
                        get[0] = i;
                        get[1] = reference_level;
                        get[2] = reference_lat;
                        get[3] = reference_lon;
                        float ref = data->data(get, 0);
                        get[0] = i;
                        get[1] = x;
                        get[2] = y;
                        get[3] = z;
                        float voxel = data->data(get,0);
                        if (!isnan(ref) && !isnan(voxel)) {
                            E_x += ref;
                            E_x_counter++;
                            E_xx += ref * ref;
                            E_xx_counter++;

                            E_y += voxel;
                            E_y_counter++;
                            E_xy += ref * voxel;
                            E_xy_counter++;
                            E_yy += voxel * voxel;
                            E_yy_counter++;
                        }
                    }
                    //Incase there were only NaN values
                    if(E_x_counter == 0) {
                      out->push_back(0.5f);
                      continue;
                    }
                    E_x /= E_x_counter;
                    E_y /= E_y_counter;
                    E_xy /= E_xy_counter;
                    E_xx /= E_xx_counter;
                    E_yy /= E_yy_counter;
                    float corr = (E_xy - (E_x * E_y)) / (sqrt(E_xx - (E_x * E_x)) * sqrt(E_yy - (E_y * E_y)));
                    out->push_back((corr+1.0)/2.0);
                }
            }
        }
        

    }
    size_t load_raw(const std::string& filename, std::vector<uint8_t>& out) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        file.seekg(0); //Need to go back to the beginning because std::ios::ate means we start at the end (so that we can get the size via tellg())

        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);
        std::vector<uint8_t> buffer_int(buffer.begin(), buffer.end());

        file.close();
        out = buffer_int;
        return fileSize;
    }

    void convert_to_float(std::vector<uint8_t>& in, std::vector<float>& out) {
        for (auto i = in.begin(); i < in.end(); i=i+4) {
            uint8_t buffer[4];
            buffer[0] = *i;
            buffer[1] = *(i + 1);
            buffer[2] = *(i + 2);
            buffer[3] = *(i + 3);
            out.push_back(*(float *)&buffer);
        }
    }

    void convert_to_double(std::vector<uint8_t>& in, std::vector<double>& out) {
        uint16_t max = 0b1111111111111111;
        double max_d = max;
        for (int i = 0; i < in.size(); i = i+2) {
            uint16_t current = in[i+1] << 8;
            current += in[i];
            double current_d = current;
            out.push_back(current_d/max_d);
        }
    }

    float max(std::vector<float>& floats) {
        float current_max = -100;
        for (auto i = floats.begin(); i < floats.end(); i++) {
            current_max = (*i > current_max) ? *i : current_max;
        }
        return current_max;
    }

    void convert_to_normalized_magnitude(std::vector<float>& floats,std::vector<float>& magnitudes) {
        for (auto i = floats.begin(); i < floats.end(); i = i + 3) {
            float x = *i;
            float y = *(i + 1);
            float z = *(i + 2);
            magnitudes.push_back(sqrtf(x*x + y*y + z*z));
        }

        //for (auto i = magnitudes.begin(); i < magnitudes.begin() + 52; i++) {
        //    std::cout << +*i << std::endl;
        //}
        float max_value = max(magnitudes);
        for (auto i = magnitudes.begin(); i < magnitudes.end(); i++) {
            *i = *i / max_value;
        }

        //for (auto i = magnitudes.begin(); i < magnitudes.begin() + 52; i++) {
        //    std::cout << +*i << std::endl;
        //}
    }



    void convert_to_byte(std::vector<float>& magnitudes, std::vector<uint8_t>& out) {
        for (auto i = magnitudes.begin(); i < magnitudes.end(); i = i + 1) {
            float f = *i*255 +0.5f;
            int value = (int)f;
            uint8_t value_byte = value;
            out.push_back(value_byte);
        }
    }


    void moveData() {
        //std::vector<uint8_t> data{};
        //std::vector<float> dataf;
        //std::vector<float> magnitudes = pearsonData;
        std::vector<float> magnitudes;
        for(uint32_t i = 0; i < dims[1];i++) {
          for(uint32_t j = 0; j < dims[2];j++) {
            for(uint32_t w = 0; w < dims[3];w++) {
              float value = res.data({(unsigned int)referencePoint[3],i,j,w}, 0);
              if(isnan(value)) {
                magnitudes.push_back(0.0f);
                continue;
              }
              magnitudes.push_back(value);
            }
          }
        }
        //size_t data_size = load_raw("D:\\MasterArbeit\\Data\\vortexStreet0000.raw", data);
        //convert_to_float(data, dataf);
        //convert_to_normalized_magnitude(dataf, magnitudes);
        std::vector<uint8_t> final_data{};
        convert_to_byte(magnitudes, final_data);
        util::memory_view<const uint8_t> upload_data(final_data);

       
        //_vertices_count = data_size / 12; //4 bytes = 1 float and 3 floats = 1 vertices
        


        //Create the image for the volume data
        constexpr VkImageAspectFlags image_aspect{ VK_IMAGE_ASPECT_COLOR_BIT };
        VkImageUsageFlags buffer_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto header_info = util::vk::initializers::imageCreateInfo(VK_FORMAT_R8_UNORM, { dims[3],dims[2],dims[1] }, buffer_usage, VK_IMAGE_TYPE_3D); 
        auto header_alloc_info = util::vma::initializers::allocationCreateInfo();
        destination = util::vk::create_image(header_info, header_alloc_info);

        //Create image view

        VkImageViewCreateInfo image_view_info = util::vk::initializers::imageViewCreateInfo(destination.image, VK_IMAGE_VIEW_TYPE_3D, VK_FORMAT_R8_UNORM);
        //ValidationLayer complains
        image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_view_info.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        image_view_info.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
        VkImageView image_view = util::vk::create_image_view(image_view_info);

        //Create sampler
        VkSamplerCreateInfo sampler_info = util::vk::initializers::samplerCreateInfo();
        VkSampler sampler = util::vk::create_sampler(sampler_info);

        //uploading image
        structures::stager::staging_image_info staging_info{};
        staging_info.transfer_dir = structures::stager::transfer_direction::upload;
        staging_info.common.data_upload = upload_data;
        staging_info.dst_image = destination.image;
        staging_info.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        staging_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        staging_info.subresource_layers.aspectMask = image_aspect;
        staging_info.image_extent = header_info.extent;
        staging_info.image_offset = { 0,0,0 };
        staging_info.bytes_per_pixel = 1;
        globals::stager.add_staging_task(staging_info);

        //VK_FORMAT_R8_UNORM
        {
            const std::string_view descriptor_id{ "data" };
            structures::uniqe_descriptor_info data_desc{ std::make_unique<structures::descriptor_info>() };
            data_desc->id = std::string(descriptor_id);
            auto descriptor_binding = util::vk::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1U);
            auto descriptor_info = util::vk::initializers::descriptorSetLayoutCreateInfo(util::memory_view(descriptor_binding));
            data_desc->layout = util::vk::create_descriptorset_layout(descriptor_info);
            auto descriptor_alloc_info = util::vk::initializers::descriptorSetAllocateInfo(globals::vk_context.general_descriptor_pool, data_desc->layout);
            auto res = vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &data_desc->descriptor_set);
            util::check_vk_result(res);
            VkDescriptorImageInfo desc_buffer_info{ sampler ,image_view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            auto write_desc_set = util::vk::initializers::writeDescriptorSet(data_desc->descriptor_set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1U, &desc_buffer_info);
            vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
            globals::descriptor_sets[descriptor_id] = std::move(data_desc);
        }
        


        //TransferFunction
        
        //set the default colors
        for (int i = 0; i < binCount; i++) {
            tf.my_color[i][0] = 1.0 * i / (binCount + 1);
            tf.my_color[i][1] = 1.0 * i / (binCount + 1);
            tf.my_color[i][2] = 1.0 * i / (binCount + 1);
            tf.my_color[i][3] = 1.0/255.0;
        }
        //Preset for optimization
        tf.my_color[0][0] = 0.0;
        tf.my_color[0][1] = 0.0;
        tf.my_color[0][2] = 1.0;

        tf.my_color[6][0] = 0.0;
        tf.my_color[6][1] = 1.0;
        tf.my_color[6][2] = 0.0;

        tf.my_color[7][0] = 1.0;
        tf.my_color[7][1] = 0.0;
        tf.my_color[7][2] = 0.0;

        tf.my_color[1][3] = 0.0;
        tf.my_color[2][3] = 0.0;
        tf.my_color[3][3] = 0.0;
        tf.my_color[4][3] = 0.0;
        tf.my_color[5][3] = 0.0;

        //Bin the alphas for the TFPicker
        for (auto i = final_data.begin(); i < final_data.end(); i = i + 1) {
            uint8_t data = *i;
            uint8_t binNumber = data * binCount / 256;
            bins[0][binNumber] += (1.0);


        }

        /* TransferFunction as Buffer
        //Now Create the Transferfunction UBO
        
        std::cout << tf.my_color[0][0] << std::endl;
        
        //Create the buffer
        VkBufferCreateInfo info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,sizeof(tf));
        VmaAllocationCreateInfo alloInfo = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
        structures::buffer_info bufferInfo = util::vk::create_buffer(info,alloInfo);
        //Upload the buffer
        structures::stager::staging_buffer_info staging_buffer_info{};
        staging_buffer_info.transfer_dir = structures::stager::transfer_direction::upload;
        staging_buffer_info.common.data_upload = util::memory_view(tf);
        staging_buffer_info.dst_buffer = bufferInfo.buffer;
        tfBuffer = bufferInfo;
        globals::stager.add_staging_task(staging_buffer_info);
        
        //Create Descriptorsets
        {
            const std::string_view descriptor_id{ "tf" };
            structures::uniqe_descriptor_info data_desc{ std::make_unique<structures::descriptor_info>() };
            data_desc->id = std::string(descriptor_id);
            auto descriptor_binding = util::vk::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 2U);
            auto descriptor_info = util::vk::initializers::descriptorSetLayoutCreateInfo(util::memory_view(descriptor_binding));
            data_desc->layout = util::vk::create_descriptorset_layout(descriptor_info);
            auto descriptor_alloc_info = util::vk::initializers::descriptorSetAllocateInfo(globals::vk_context.general_descriptor_pool, data_desc->layout);
            auto res = vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &data_desc->descriptor_set);
            util::check_vk_result(res);
            VkDescriptorBufferInfo desc_buffer_info{bufferInfo.buffer,0,VK_WHOLE_SIZE };
            auto write_desc_set = util::vk::initializers::writeDescriptorSet(data_desc->descriptor_set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2U, &desc_buffer_info);
            vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
            globals::descriptor_sets[descriptor_id] = std::move(data_desc);
        }
        */

        //Transfer function as Texture
        
        constexpr VkFormat image_format{ VK_FORMAT_R32G32B32A32_SFLOAT };
        auto image_info = util::vk::initializers::imageCreateInfo(image_format, {binCount, 1, 1 }, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,VK_IMAGE_TYPE_1D);
        auto alloc_info = util::vma::initializers::allocationCreateInfo();
        auto [image, view] = util::vk::create_image_with_view(image_info, alloc_info);

        // uploading image
        structures::stager::staging_image_info image_staging{};
        image_staging.dst_image = image.image;
        image_staging.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_staging.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_staging.subresource_layers.aspectMask = image_aspect;
        image_staging.bytes_per_pixel = 4*4;
        image_staging.image_extent.width = binCount;
        image_staging.image_extent.height = 1;
        image_staging.image_extent.depth = 1;
        image_staging.common.data_upload = util::memory_view(tf);
        tfImage = image;
        globals::stager.add_staging_task(image_staging);

        // creating descriptor set + layout for heat_map
        {
            const std::string_view descriptor_id{ "tf" };
            structures::uniqe_descriptor_info tf_desc{ std::make_unique<structures::descriptor_info>() };
            tf_desc->id = std::string(descriptor_id);
            auto tf_binding = util::vk::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,2);
            auto descriptor_info = util::vk::initializers::descriptorSetLayoutCreateInfo(tf_binding);
            tf_desc->layout = util::vk::create_descriptorset_layout(descriptor_info);
            auto descriptor_alloc_info = util::vk::initializers::descriptorSetAllocateInfo(globals::vk_context.general_descriptor_pool, tf_desc->layout);
            vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &tf_desc->descriptor_set);
            VkDescriptorImageInfo desc_image_info{ globals::persistent_samplers.get(util::vk::initializers::samplerCreateInfo(VK_FILTER_LINEAR)), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            auto write_desc_set = util::vk::initializers::writeDescriptorSet(tf_desc->descriptor_set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &desc_image_info);
            vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
            globals::descriptor_sets[tf_desc->id] = std::move(tf_desc);
        }

        globals::stager.wait_for_completion();    //Wait for completion
    }

    float max(float* arr, int arrSize) {
        float biggestValue = 0;
        for (int i = 0; i < arrSize;i++) {
            if (biggestValue < arr[i]) {
                biggestValue = arr[i];
            }
        }
        return biggestValue;
    }

    void plotHistogram(ImVec2 pos, ImVec2 size,int currentSelected) {
        /*
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        for (int i = 0; i < binCount; i++)
        {
            ImVec4 barColor = ImVec4(tf.my_color[i][0], tf.my_color[i][1], tf.my_color[i][2], 1.0);
            ImVec2 barSize(size.x, (bins[i] / max(bins, binCount)) * size.y);
            //ImVec4 barColor = ImVec4(1,1,1,1);
            ImVec2 barLocal = ImVec2((i * size.x),(size.y - barSize.y));
            ImVec2 barPos = ImVec2(pos.x + barLocal.x,pos.y + barLocal.y);
            drawList->AddRectFilled(barPos,ImVec2(barPos.x + barSize.x,pos.y+size.y), ImGui::ColorConvertFloat4ToU32(barColor));
        }
        */
        ImGui::PlotHistogram("My Histogram",bins[currentSelected],8,0,NULL,1.0,FLT_MAX,size,4);
    }

    void calculateColorOfPos(int x,float* col) {
        //Get dot to the left and to the right
        Dot leftDot = {ImVec2(-1,-1)}; //The dot to the left of the position
        Dot rightDot = {ImVec2(-1,-1)}; //The dot to the right of the position
        Dot exactDot = { ImVec2(-1,-1) }; //If there is a dot exactly at the x position then it will be stored here
        for (int i = 0; i < gradientDots.size();i++) {
            Dot d = gradientDots[i];
            if (d.pos.x == x) { //If there is a dot exactly at the pos
                exactDot = d;
                break;
            }
            if (d.pos.x > x) { // If the dot is to the right of the x then we found the left and right Dot (The dots are sorted in the vector)
                rightDot = d;
                break;
            }
            leftDot = d;
        }
        //In case no dot was found to the left or to the right
        if (leftDot.pos.x == -1) {
            leftDot = gradientDots[0];
        }
        if (rightDot.pos.x == -1) {
            rightDot = gradientDots[gradientDots.size()-1];
        }

        //If there is a dot at the exact position then return that dot color
        if (exactDot.pos.x != -1) {
            col[0] = exactDot.color[0];
            col[1] = exactDot.color[1];
            col[2] = exactDot.color[2];
            return;
        }
        //Else interpolate
        float maxD = rightDot.pos.x - leftDot.pos.x;
        float relD = x - leftDot.pos.x; 
        float ratio = relD / maxD;
        col[0] = (rightDot.color[0] - leftDot.color[0]) * ratio + leftDot.color[0];
        col[1] = (rightDot.color[1] - leftDot.color[1]) * ratio + leftDot.color[1];
        col[2] = (rightDot.color[2] - leftDot.color[2]) * ratio + leftDot.color[2];
    }

    float calculateAlphaOfPos(int x) {
        //Get dot to the left and to the right
        Dot leftDot = { ImVec2(-1,-1) }; //The dot to the left of the position
        Dot rightDot = { ImVec2(-1,-1) }; //The dot to the right of the position
        Dot exactDot = { ImVec2(-1,-1) }; //If there is a dot exactly at the x position then it will be stored here
        for (int i = 0; i < histogramDots.size() ; i++) {
            Dot d = histogramDots[i];
            if (d.pos.x == x) { //If there is a dot exactly at the pos
                exactDot = d;
                break;
            }
            if (d.pos.x > x) { // If the dot is to the right of the x then we found the left and right Dot (The dots are sorted in the vector)
                rightDot = d;
                break;
            }
            leftDot = d;
        }
        //In case no dot was found to the left or to the right
        if (leftDot.pos.x == -1) {
            leftDot = histogramDots[0];
        }
        if (rightDot.pos.x == -1) {
            rightDot = histogramDots[histogramDots.size() - 1];
        }

        //If there is a dot at the exact position then return that dot color
        if (exactDot.pos.x != -1) {
            return (100-exactDot.pos.y)/100;
        }
        //Else interpolate
        float maxD = rightDot.pos.x - leftDot.pos.x +1;
        float relD = x - leftDot.pos.x;
        float ratio = relD / maxD;
        float alpha = (100 - leftDot.pos.y) + ratio * (100.0 - rightDot.pos.y);
        if (alpha < 0) {
            alpha = 0;
        }
        return alpha/10000.0;
    }

    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;
    std::vector<std::vector<float>> g;
    int histogramDotsPos = 0;
    int gradientDotsPos = 0;
    int colorPos = 0;
    int stepCount = 1;
    float stepSizeHistogram = 5.0f;
    float stepSizeGradient = 2.0f / 255.0f;

    void volume_workbench::calculateTF(structures::image_info image_info,size_t width, size_t height) {
        bool adam = true;
        std::vector<uint8_t> tmpImage = std::vector<uint8_t>(width*height*8);
        std::vector<double> currentImage;
        //Get current Image
        VkImage frame = image_info.image;

        structures::stager::staging_image_info staging_image_info{};
        staging_image_info.transfer_dir = structures::stager::transfer_direction::download; // download
        staging_image_info.common.data_download = util::memory_view(tmpImage);
        staging_image_info.image_extent.width = width;
        staging_image_info.image_extent.height = height;
        staging_image_info.image_extent.depth = 1;
        staging_image_info.dst_image = frame;
        staging_image_info.start_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        staging_image_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        staging_image_info.subresource_layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        staging_image_info.bytes_per_pixel = 8;
        globals::stager.add_staging_task(staging_image_info);
        globals::stager.wait_for_completion();
        //Convert bytes to doubles and store in referenceImage
        convert_to_double(tmpImage, currentImage);
        /*
        if (error == -1) {
            std::cout << referenceImage.size() << std::endl;
            std::ofstream out("out3.txt");
            std::streambuf* coutbuf = std::cout.rdbuf();
            std::cout.rdbuf(out.rdbuf());
            std::cout.precision(17);
            /*
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    for (int w = 0; w < 4; w++) {
                        double currentByte = referenceImage[(4 * height * i) + j * 4 + w];
                        std::cout << std::fixed << currentByte << std::endl;
                    }

                }
            }
            
            
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    for (int w = 0; w < 4; w++) {
                        double currentByte = currentImage[(4 * height * i) + j * 4 + w];
                        std::cout << std::fixed << currentByte << std::endl;
                    }

                }
            }
            
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    for (int w = 0; w < 4; w++) {
                        double referenceValue = referenceImage[(4 * width * j) + i * 4 + w];
                        double currentValue = currentImage[(4 * width * j) + i * 4 + w];
                        std::cout << std::fixed << ((referenceValue - currentValue)+1.0)/2.0 << std::endl;
                    }
                }
            }
            
            std::cout.rdbuf(coutbuf); //reset to standard output again
        }
        */
        //Calculate Error
        double current_error = 0.0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                float dist[4];
                for (int w = 0; w < 4; w++) {
                    double referenceValue = referenceImage[(4 * width * j) + i * 4 + w];
                    double currentValue = currentImage[(4 * width * j) + i * 4 + w];
                    dist[w] = referenceValue - currentValue;
                }
                current_error += dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] + dist[3] * dist[3];
            }
        }
        current_error /= (height * (width)); //MSE
        current_error *= 10;
        
        if (error != -1.0 && histogramDotsPos == 0 && gradientDotsPos == 0 && colorPos == 0) {
            error = current_error;
            std::cout << "New Error: " << error << std::endl;
        }
        
        
        //If there is no previous error then we are at the first step
        if (error == -1.0) {
            error = current_error;
            //Also create the m and v vectors
            m.clear();
            v.clear();
            g.clear();
            for (int i = 0; i < histogramDots.size();i++) {
                m.push_back({0,0,0,0}); //R,G,B,Y(Y position im histogram) format
                v.push_back({0,0,0,0}); //R,G,B,Y(Y position im histogram) format
                g.push_back({ 0,0,0,0 }); //R,G,B,Y(Y position im histogram) format
            }
            for (int i = 0; i < gradientDots.size();i++) {
                m.push_back({ 0,0,0,0 }); //R,G,B,Y(Y position im histogram) format
                v.push_back({ 0,0,0,0 }); //R,G,B,Y(Y position im histogram) format
                g.push_back({ 0,0,0,0 }); //R,G,B,Y(Y position im histogram) format
            }
            return;
        }
        //Calculate gradient (Disgusting code, but humor me)
        for (int i = histogramDotsPos; i <= histogramDots.size();i++) {
            //Calculate gradient for previous dot
            if (i > 0) { 
                histogramDots[i-1].pos.y += 1;
                float deltaError = (error - current_error)/(-1);
                g[i - 1][3] = deltaError;
            }
            histogramDotsPos++;
            //Create change, in histogram dots only the y value needs to change
            if (i == histogramDots.size()) {
                break;
            }
            histogramDots[i].pos.y -= 1;
            /*
            for(int j = 0; j < histogramDots.size();j++) {
              std::cout << "J: " << j << " Pos: " << histogramDots[j].pos.y << std::endl;
            }
            */
            return;
        }
        float gradientStep = (1.0f / 255.0f);
        for (int i = gradientDotsPos; i <= gradientDots.size();i++) {
            for (int c = colorPos; c < 3;c++) {
                if (c > 0) {
                    gradientDots[i].color[c-1] -= gradientStep;
                    float deltaError = (error - current_error)/ gradientStep;
                    g[histogramDots.size() + i][(c - 1)] = deltaError;
                }
                else if(i > 0) {
                    gradientDots[i-1].color[2] -= gradientStep;
                    float deltaError = (error - current_error) / gradientStep;
                    g[histogramDots.size() + i - 1][2] = deltaError;
                }
                if (i == gradientDots.size()) {
                    break;
                }
                gradientDots[i].color[c] += gradientStep;
                colorPos++;

                return;
            }
            gradientDotsPos++;
            if (i == gradientDots.size()) {
                break;
            }
            colorPos = 0;
            return;
        }
        
        //ADAM
        //Calculate m and v
        if(adam) {


          float beta1 = 0.99f;
          float beta2 = 0.99f;

          for(int i = 0; i < m.size(); i++) {
            for(int j = 0; j < m[i].size(); j++) {
              m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
              v[i][j] = beta2 * v[i][j] + (1 - beta2) * g[i][j] * g[i][j];
            }
          }

          //Final persistent change

          for(int i = 0; i < histogramDots.size(); i++) {
              //Bias correction
            float bc_m = m[i][3] / (1 - powf(beta1, stepCount));
            float bc_v = v[i][3] / (1 - powf(beta2, stepCount));

            histogramDots[i].pos.y = histogramDots[i].pos.y - stepSizeHistogram * (bc_m / (sqrtf(bc_v) + 0.0000001));
            std::cout << "i: " << i << " Gradient: " << g[i][3] << " Difference: " << stepSizeHistogram * (bc_m / (sqrtf(bc_v) + 0.0000001)) << std::endl;
            //Constraints
            if(histogramDots[i].pos.y < 4) {
              histogramDots[i].pos.y = 4;
            } else if(histogramDots[i].pos.y > 102) {
              histogramDots[i].pos.y = 102;
            }
          }
          for(int i = 0; i < gradientDots.size(); i++) {
            for(int c = 0; c < 3; c++) {
                //Bias correction
              float bc_m = m[histogramDots.size() + i][c] / (1 - powf(beta1, stepCount));
              float bc_v = v[histogramDots.size() + i][c] / (1 - powf(beta2, stepCount));
              //std::cout << "i: " << i << " bc_m: " << bc_m << " bc_v" << bc_v << std::endl;
              gradientDots[i].color[c] = gradientDots[i].color[c] - stepSizeGradient * (bc_m / (sqrtf(bc_v) + 0.0000001));
              //std::cout << "i: " << i << " Difference: " << stepSizeGradient * (bc_m / (sqrtf(bc_v) + 0.000001)) << std::endl;
              if(gradientDots[i].color[c] < 0.0f) {
                gradientDots[i].color[c] = 0.0f;
              } else if(gradientDots[i].color[c] > 1.0f) {
                gradientDots[i].color[c] = 1.0f;
              }
            }

          }
        }
        else { //Gradient descent
          for(int i = 0; i < histogramDots.size(); i++) {

            histogramDots[i].pos.y = histogramDots[i].pos.y - stepSizeHistogram * g[i][3];
            std::cout << "I: " << i << " G: " << g[i][3] << std::endl;
            //Constraints
            if(histogramDots[i].pos.y < 4) {
              histogramDots[i].pos.y = 4;
            } else if(histogramDots[i].pos.y > 102) {
              histogramDots[i].pos.y = 102;
            }
          }
          for(int i = 0; i < gradientDots.size(); i++) {
            for(int c = 0; c < 3; c++) {
              gradientDots[i].color[c] = gradientDots[i].color[c] - stepSizeGradient * g[histogramDots.size() + i][c];
              if(gradientDots[i].color[c] < 0.0f) {
                gradientDots[i].color[c] = 0.0f;
              } else if(gradientDots[i].color[c] > 1.0f) {
                gradientDots[i].color[c] = 1.0f;
              }
            }

          }

        }


        std::cout << "Step: " << stepCount << " Error: " << error << std::endl;
        stepCount++;

        //Reset everything
        histogramDotsPos = 0;
        gradientDotsPos = 0;
        colorPos = 0;
        return;



    }
    
    
    void volume_workbench::show() {
        if (!active)
            return;

        std::string path;
        ImGui::Begin("Load Data");

        //List with correlation algorithms
        std::vector<const char*> itemsUnfinished = {"Dataset", "Pearson", "Spearmen", "Kendall Rank", "KSG"};
        std::vector<const char*> itemsFinished = {"Dataset", "Pearson (Calculated)", "Spearmen (Calculated)", "Kendall Rank (Calculated)", "KSG (Calculated)" };
        std::vector<const char*> items;
        items.push_back(itemsFinished[0]);
        if (pearsonData.size() == 0) {
            items.push_back(itemsUnfinished[1]);
        }
        else {
            items.push_back(itemsFinished[1]);
        }
        if (spearmenData.size() == 0) {
            items.push_back(itemsUnfinished[2]);
        }
        else {
            items.push_back(itemsFinished[2]);
        }
        if (kendallData.size() == 0) {
            items.push_back(itemsUnfinished[3]);
        }
        else {
            items.push_back(itemsFinished[3]);
        }
        if (ksgData.size() == 0) {
            items.push_back(itemsUnfinished[4]);
        }
        else {
            items.push_back(itemsFinished[4]);
        }

        static const char* current_item = "Dataset";
        static const char* reference_item = "Dataset";
        bool changed_selection = false;


        if (ImGui::BeginCombo("Correlation Algorithms", current_item)) {
            for (int i = 0; i < items.size();i++) {
                bool is_selected = (current_item == items[i]);
                if (ImGui::Selectable(items[i], is_selected)) {
                    current_item = items[i];
                    changed_selection = true;
                }
                if (is_selected) {
                    
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if(ImGui::SliderInt("Reference member (for Dataset visualization)", &referencePoint[3], 0, 99)) {
          changed_selection = true;


        }
        if (ImGui::Button("Pick current choice as reference")) {
            if (current_item == itemsFinished[1] || current_item == itemsFinished[2] || current_item == itemsFinished[3] || current_item == itemsFinished[4]) {
                reference_item = current_item;

                std::vector<uint8_t> tmpImage = std::vector<uint8_t>(plot_data.read().width * plot_data.read().height * 8);
                //Get reference Image

                VkImage frame = plot_data.read().image.image;

                structures::stager::staging_image_info staging_image_info{};
                staging_image_info.transfer_dir = structures::stager::transfer_direction::download; // download
                staging_image_info.common.data_download = util::memory_view(tmpImage);
                staging_image_info.image_extent.width = plot_data.read().width;
                staging_image_info.image_extent.height = plot_data.read().height;
                staging_image_info.image_extent.depth = 1;
                staging_image_info.dst_image = frame;
                staging_image_info.start_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                staging_image_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                staging_image_info.subresource_layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                staging_image_info.bytes_per_pixel = 8;
                globals::stager.add_staging_task(staging_image_info);
                globals::stager.wait_for_completion();
                //Convert bytes to doubles and store in referenceImage
                convert_to_double(tmpImage, referenceImage);

            }
        }

        ImGui::Text(reference_item);

        if (ImGui::Button("Optimize current selection to reference")) {
            optimize = !optimize;
        }
        if (optimize) {
            if (current_item != reference_item && (current_item == itemsFinished[1] || current_item == itemsFinished[2] || current_item == itemsFinished[3] || current_item == itemsFinished[4])) {
                calculateTF(plot_data.read().image, plot_data.read().width, plot_data.read().height);
            }
        }

        //ReferencePoint text fields
        ImGui::InputInt3("Reference Point (Latitude, Longitude, Level)",referencePoint);
        //Button to start the calculation
        if (ImGui::Button("Calculate")) {
            if (current_item == itemsUnfinished[1]) {
                auto start = std::chrono::high_resolution_clock::now();
                pearson_correlation(&res, referencePoint[0], referencePoint[1], referencePoint[2], &pearsonData);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "Pearson took: " << duration.count() << " milliseconds" << std::endl;
                changed_selection = true;
                current_item = itemsFinished[1];
                for(auto i = pearsonData.begin(); i < pearsonData.end(); i = i + 1) {
                  float data = *i;
                  uint8_t binNumber = data * binCount;
                  bins[1][binNumber] += (1.0);
                  //std::cout << "Bin number: " << binNumber << std::endl;
                }

            }
            else if (current_item == itemsUnfinished[2]) {
                auto start = std::chrono::high_resolution_clock::now();
                spearmen_correlation(&res, referencePoint[0], referencePoint[1], referencePoint[2], &spearmenData);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "Spearman took: " << duration.count() << " milliseconds" << std::endl;
                changed_selection = true;
                current_item = itemsFinished[2];
                for(auto i = spearmenData.begin(); i < spearmenData.end(); i = i + 1) {
                  float data = *i;
                  uint8_t binNumber = data * binCount;
                  bins[2][binNumber] += (1.0);
                  //std::cout << "Bin number: " << binNumber << std::endl;
                }
            }
            else if (current_item == itemsUnfinished[3]) {
                auto start = std::chrono::high_resolution_clock::now();
                kendall_rank_correlation(&res, referencePoint[0], referencePoint[1], referencePoint[2], &kendallData);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "Kendall took: " << duration.count() << " milliseconds" << std::endl;
                changed_selection = true;
                current_item = itemsFinished[3];
                for(auto i = kendallData.begin(); i < kendallData.end(); i = i + 1) {
                  float data = *i;
                  if(data > 0.0f && data < 1.0f) {
                    //std::cout << data << std::endl;
                    uint8_t binNumber = data * binCount;
                    //std::cout << int(binNumber) << std::endl;
                    bins[3][binNumber] += (1.0);
                  }
                }
            }
            else if(current_item == itemsUnfinished[4]) {
                auto start = std::chrono::high_resolution_clock::now();
                KSG_mutual_information(&res, referencePoint[0], referencePoint[1], referencePoint[2], &ksgData);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "KSG took: " << duration.count() << " milliseconds" << std::endl;
                changed_selection = true;
                current_item = itemsFinished[4];
                //Find biggest value
                float biggest = 0.0f;
                float smallest = 10.0f;
                for(int i = 0; i < ksgData.size();i++) {
                  if(ksgData[i] > biggest) {
                    biggest = ksgData[i];
                  }
                  if(ksgData[i] < smallest) {
                    smallest = ksgData[i];
                  }
                }
                std::cout << "Biggest: " << biggest << std::endl;
                std::cout << "Smallest: " << smallest << std::endl;
                //normalize the data to the range [0,1]
                for(int i = 0; i < ksgData.size();i++) {
                  ksgData[i] = (ksgData[i] / biggest);
                }
                biggest = 0.0f;
                for(int i = 0; i < ksgData.size(); i++) {
                  if(ksgData[i] > biggest) {
                    biggest = ksgData[i];
                  }
                }
                std::cout << "Biggest: " << biggest << std::endl;
                for(auto i = ksgData.begin(); i < ksgData.end(); i = i + 1) {
                    float data = *i;
                    
                    if(data > 0.0f && data < 1.00001f) {

                      //std::cout << data << std::endl;
                      uint8_t binNumber = (data) * binCount;
                      //std::cout << int(binNumber) << std::endl;
                      bins[4][binNumber] += (1.0);
                    }
                }
                for(int i = 0; i < binCount;i++) {
                  std::cout << "bin " << i << " = " << bins[4][i] << std::endl;

                }
            }
        }

        ImGui::Image(plot_data.read().image_descriptor, ImGui::GetContentRegionAvail());

        ImGui::Begin("TFPicker");
        ImVec4 oldBG = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        //ImGui::GetStyle().Colors[ImGuiCol_ChildBg] = ImVec4(1.0, 0, 0, 1.0);

        ImGui::BeginChild("Histogram",ImVec2(ImGui::GetWindowSize().x,100));
        //ImGui::PlotHistogram("", bins, binCount, 0, "Alphas", FLT_MAX, FLT_MAX, ImVec2(0, 100));
        if(current_item == itemsFinished[0]) {
          plotHistogram(ImGui::GetItemRectMin(), ImVec2(ImGui::GetItemRectSize().x, 100), 0);
        }
        else if(current_item == itemsFinished[1]) {
          plotHistogram(ImGui::GetItemRectMin(), ImVec2(ImGui::GetItemRectSize().x, 100),1);
        } else if(current_item == itemsFinished[2]) {
          plotHistogram(ImGui::GetItemRectMin(), ImVec2(ImGui::GetItemRectSize().x, 100), 2);
        } else if(current_item == itemsFinished[3]) {
          plotHistogram(ImGui::GetItemRectMin(), ImVec2(ImGui::GetItemRectSize().x, 100), 3);
        }
        else if(current_item == itemsFinished[4]) {
          plotHistogram(ImGui::GetItemRectMin(), ImVec2(ImGui::GetItemRectSize().x, 100), 4);
        }
        
        ImVec2 mousePos = ImGui::GetItemRectMin();
        ImVec2 windowPos = ImGui::GetWindowPos();

        //If first frame then add the predefined dots:
        if (histogramDots.size() < binCount) {
            int width = (ImGui::GetWindowSize().x - 30) - 5;
            int step = width / binCount;
            for (int i = 0; i <= binCount;i++) {
                Dot dot = {ImVec2(5+(i*step),90)};
                if (i == 0 || i == binCount-1) {
                    dot.isXLocked = true;
                }
                histogramDots.push_back(dot);
            }
            /*
            Dot left = { ImVec2(5,100) };
            left.isXLocked = true;
            histogramDots.push_back(left);

            Dot right = { ImVec2(ImGui::GetWindowSize().x-30,0) };
            right.isXLocked = true;
            histogramDots.push_back(right);
            */
        }
         
        
        //std::cout << my_color[0][0] << std::endl;
        

        //DrawHistogramWithCustomColors("",bins,binCount,10,100);

        //OnMouseClick
        //ImGui::IsMouseDown
        
        if(ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            ImVec2 mousePos = ImGui::GetMousePos();
            ImVec2 histogramPos = ImGui::GetItemRectMin();
            ImVec2 histogramSize = ImGui::GetItemRectSize();
            //mousePos.x -= histogramPos.x;
            //mousePos.y -= histogramPos.y;
            //selectedBin = mousePos.x / (histogramSize.x / binCount);
            bool dragged = false;
            for (auto& dot : histogramDots){
                if (ImGui::IsMouseHoveringRect(ImVec2((dot.pos.x+windowPos.x) -10, (dot.pos.y+windowPos.y) -10),ImVec2((dot.pos.x + windowPos.x) + 10, (dot.pos.y + windowPos.y) +10)))
                {
                    dot.isDragging = true;
                    dragged = true;
                    break;
                }

            }
            if (!dragged) {
                ImVec2 position = ImVec2(mousePos.x - windowPos.x,mousePos.y - windowPos.y);
                Dot d = Dot(position);
                histogramDots.push_back(d);
                //Sort dot list by x position
                std::sort(histogramDots.begin(), histogramDots.end(), dotSortFunction);

            }
            
        }
        
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            for (auto& dot : histogramDots){
                if (dot.isDragging) {
                    
                    ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    ImGui::ResetMouseDragDelta();
                    if (!dot.isXLocked) {
                        dot.pos.x += delta.x;
                        if (dot.pos.x > ImGui::GetItemRectSize().x-125) {
                            dot.pos.x = ImGui::GetItemRectSize().x-125;
                        }
                        else if (dot.pos.x < 10) {
                            dot.pos.x = 10;
                        }
                    }
                    dot.pos.y += delta.y;
                    if (dot.pos.y > ImGui::GetItemRectSize().y+2) {
                        dot.pos.y = ImGui::GetItemRectSize().y+2;
                    }
                    else if (dot.pos.y < 0) {
                        dot.pos.y = 0;
                    }
                    std::sort(histogramDots.begin(), histogramDots.end(), dotSortFunction);
                }

            }
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            for (auto& dot : histogramDots) {
                dot.isDragging = false;
            }
        }
        
        Dot prev = {ImVec2(-1,-1)};
        for (auto& dot : histogramDots){
            ImDrawList* drawList = ImGui::GetForegroundDrawList();
            ImVec2 position = ImVec2(dot.pos.x + windowPos.x,dot.pos.y + windowPos.y);
            drawList->AddCircleFilled(position,5.0f,IM_COL32(155,155,155,255),16);
            //Draw lines connecting them
            if (prev.pos.x != -1) {
                ImU32 lineColor = IM_COL32(155,155,155,255);
                ImVec2 prevPos = ImVec2(prev.pos.x + windowPos.x,prev.pos.y + windowPos.y);
                ImVec2 dotPos = ImVec2(dot.pos.x + windowPos.x, dot.pos.y + windowPos.y);
                drawList->AddLine(prevPos, dotPos,lineColor,2.0f);
            }
            prev = dot;
        }
        

        ImGui::EndChild();
        ImGui::BeginChild("ColorGradient", ImVec2(ImGui::GetWindowSize().x, 500));
        //Draw Colorpicker
        
        ImGui::InvisibleButton("Test",ImVec2(ImGui::GetWindowSize().x, 50)); //Important, else Area height collapses to 0 and it stops working

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        //If first frame then add the leftmost and rightmost dots
        if (gradientDots.size() < binCount) {
            int width = (ImGui::GetWindowSize().x - 30) - 5;
            int step = width / binCount;
            for (int i = 0; i <= binCount;i++) {
                Dot dot = {ImVec2(5+(i*step),130)};
                dot.isYLocked = true;
                dot.color[0] = 0.5f;
                dot.color[1] = 0.5f;
                dot.color[2] = 0.5f;
                if (i == 0 || i == binCount-1) {
                    dot.isXLocked = true;
                }
                gradientDots.push_back(dot);
            }
            /*
            Dot left = {ImVec2(0,130)};
            left.isYLocked = true;
            left.isXLocked = true;
            left.color[0] = 1.0f;
            left.color[1] = 1.0f;
            left.color[2] = 0.0f;
            gradientDots.push_back(left);

            Dot right = { ImVec2(ImGui::GetWindowSize().x-30,130) };
            right.isYLocked = true;
            right.isXLocked = true;
            right.color[0] = 0.0f;
            right.color[1] = 1.0f;
            right.color[2] = 0.0f;
            gradientDots.push_back(right);
            */
        }

        //Draw Area
        prev = {ImVec2(-1,-1)};
        for (auto dot : gradientDots) {
            
            if (prev.pos.x != -1) {
                ImVec2 rectTopLeft = ImVec2(prev.pos.x+windowPos.x,100+windowPos.y);
                ImVec2 rectBottomRight = ImVec2(dot.pos.x + windowPos.x,150 + windowPos.y);
                ImU32 leftCol = IM_COL32(prev.color[0] * 255, prev.color[1] * 255, prev.color[2] * 255,255);
                ImU32 rightCol = IM_COL32(dot.color[0] * 255, dot.color[1] * 255, dot.color[2] * 255, 255);

                drawList->AddRectFilledMultiColor(rectTopLeft, rectBottomRight, leftCol,rightCol,rightCol,leftCol);
            }
            prev = dot;
        }
        //Draw Dots
        for (auto dot : gradientDots) {
            ImVec2 position = ImVec2(dot.pos.x + windowPos.x, dot.pos.y + windowPos.y);
            drawList->AddCircleFilled(position, 5.0f, IM_COL32(155, 155, 155, 255), 16);
        }

        
        
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            ImVec2 mousePos = ImGui::GetMousePos();
            ImVec2 gradientPos = ImGui::GetItemRectMin();
            ImVec2 gradientSize = ImGui::GetItemRectSize();
            //mousePos.x -= histogramPos.x;
            //mousePos.y -= histogramPos.y;
            //selectedBin = mousePos.x / (histogramSize.x / binCount);
            bool dragged = false;
            for (auto& dot : gradientDots) {
                if (ImGui::IsMouseHoveringRect(ImVec2((dot.pos.x + windowPos.x) - 10, (dot.pos.y + windowPos.y) - 10), ImVec2((dot.pos.x + windowPos.x) + 10, (dot.pos.y + windowPos.y) + 10)))
                {
                    dot.isDragging = true;
                    for (auto& dot2: gradientDots) {
                        dot2.isSelected = false;
                    }
                    dot.isSelected = true;
                    dragged = true;
                    break;
                }

            }
            if (!dragged) {
                ImVec2 position = ImVec2(mousePos.x - windowPos.x, 130);
                Dot d = Dot(position);
                for (auto& dot2 : gradientDots) {
                    dot2.isSelected = false;
                }
                d.isYLocked = true;
                d.isSelected = true;
                d.color[0] = 0.5f;
                d.color[1] = 0.5f;
                d.color[2] = 0.5f;
                gradientDots.push_back(d);
                //Sort dot list by x position
                std::sort(gradientDots.begin(), gradientDots.end(), dotSortFunction);

            }

        }
        

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            for (auto& dot : gradientDots) {
                if (dot.isDragging) {

                    ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    ImGui::ResetMouseDragDelta();
                    if (!dot.isXLocked) {
                        dot.pos.x += delta.x;
                    }
                    if (!dot.isYLocked) {
                        dot.pos.y += delta.y;
                    }
                    
                    std::sort(gradientDots.begin(), gradientDots.end(), dotSortFunction);
                }

            }
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            for (auto& dot : gradientDots) {
                dot.isDragging = false;
            }
        }

        for (auto& dot : gradientDots) {
            if (dot.isSelected) {
                ImGui::ColorEdit3("Dot Color", dot.color);
                
            }
        }
        
        ImGui::EndChild();
        //ImGui::ColorEdit4("Color", tf.my_color[selectedBin]);
        
        ImGui::GetStyle().Colors[ImGuiCol_ChildBg] = oldBG;
        ImGui::End();
        
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_A)) {
            cameraYAngle-= 0.1f;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(SDL_SCANCODE_D) ) {
            cameraYAngle += 0.1f;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_S)) {
            cameraZAngle -= 0.1f;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) || ImGui::IsKeyPressed(ImGuiKey_W)) {
            cameraZAngle += 0.1f;
        }
        //calculateTF(plot_data.read().image, plot_data.read().width, plot_data.read().height);
        
        
        //util::vma::upload_data(util::memory_view(tf),tfBuffer);
        /*
        {
            const std::string_view descriptor_id{ "tf" };
            //auto res = vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &data_desc->descriptor_set);
            VkDescriptorBufferInfo desc_buffer_info{ tfBuffer.buffer,0,VK_WHOLE_SIZE };
            auto write_desc_set = util::vk::initializers::writeDescriptorSet(globals::descriptor_sets[descriptor_id]->descriptor_set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2U, &desc_buffer_info);
            vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
            globals::descriptor_sets[descriptor_id]->descriptor_set;
        }
        */

        if (changed_selection) {
            std::vector<float> magnitudes;

            if(current_item == itemsFinished[0]) {

              for(uint32_t i = 0; i < dims[1]; i++) {
                for(uint32_t j = 0; j < dims[2]; j++) {
                  for(uint32_t w = 0; w < dims[3]; w++) {
                    float value = res.data({(unsigned int)referencePoint[3],i,j,w}, 0);
                    if(isnan(value)) {
                      magnitudes.push_back(0.01f);
                      continue;
                    }
                    magnitudes.push_back(value);
                  }
                }
              }

            }
            else if (current_item == itemsFinished[1]) {
                magnitudes = pearsonData;
            }
            else if (current_item == itemsFinished[2]) {
                magnitudes = spearmenData;
            }
            else if (current_item == itemsFinished[3]) {
                magnitudes = kendallData;
            }
            else if(current_item == itemsFinished[4]){
                magnitudes = ksgData;
            }

            std::vector<uint8_t> final_data{};
            convert_to_byte(magnitudes, final_data);
            util::memory_view<const uint8_t> upload_data(final_data);

            auto header_info = util::vk::initializers::imageCreateInfo(VK_FORMAT_R8_UNORM, { dims[3],dims[2],dims[1] }, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TYPE_3D);

            //uploading image
            structures::stager::staging_image_info staging_info{};
            staging_info.transfer_dir = structures::stager::transfer_direction::upload;
            staging_info.common.data_upload = upload_data;
            staging_info.dst_image = destination.image;
            staging_info.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            staging_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            staging_info.subresource_layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            staging_info.image_extent = header_info.extent;
            staging_info.image_offset = { 0,0,0 };
            staging_info.bytes_per_pixel = 1;
            globals::stager.add_staging_task(staging_info);
            globals::stager.wait_for_completion();

            //changed_selection = false;
        }

        //Update tf
        float width = 1900.0f / binCount;
        for (int i = 0; i < binCount;i++) {
            float color[3]; 
            calculateColorOfPos(i*width + width*0.5, color);
            float alpha = calculateAlphaOfPos(i * width + width * 0.5);
            tf.my_color[i][0] = color[0];
            tf.my_color[i][1] = color[1];
            tf.my_color[i][2] = color[2];
            tf.my_color[i][3] = alpha;
            if(changed_selection) {
              std::cout << "Bin " << i * width + width * 0.5f << ": " << color[0] << "//" << color[1] << "//" << color[2] << "//" << alpha << "//" << std::endl;
              
            }
            //std::cout << "Bin " << i*width+width*0.5f << ": " << color[0] << "//" << color[1] << "//" << color[2] << "//" << alpha << "//" << std::endl;
        }
        changed_selection = false;
        structures::stager::staging_image_info staging_image_info{};
        staging_image_info.transfer_dir = structures::stager::transfer_direction::upload; // download
        staging_image_info.common.data_upload = util::memory_view(tf);
        staging_image_info.dst_image = tfImage.image;
        staging_image_info.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        staging_image_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        staging_image_info.subresource_layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        staging_image_info.bytes_per_pixel = 4 * 4;
        staging_image_info.image_extent.width = binCount;
        staging_image_info.image_extent.height = 1;
        staging_image_info.image_extent.depth = 1;
        globals::stager.add_staging_task(staging_image_info);
        /*
        structures::stager::staging_buffer_info staging_buffer_info{};
        staging_buffer_info.transfer_dir = structures::stager::transfer_direction::upload;
        staging_buffer_info.common.data_upload = util::memory_view(tf);
        staging_buffer_info.dst_buffer = tfBuffer.buffer;
        globals::stager.add_staging_task(staging_buffer_info);
        */
        globals::stager.wait_for_completion();
        render_plot();
        
        //ImGui::GetIO().ConfigFlags = old;
        //ImGui::GetIO().WantCaptureKeyboard = false;
        ImGui::End();
    }

    void volume_workbench::render_plot()
    {
        if (logger.logging_level >= logging::level::l_5)
            logger << logging::info_prefix << " volume::render_plot()" << logging::endl;
        pipelines::volume_renderer::render_info render_info{
            *this,  // workbench (is not changed, the renderer only reads information)
            {},     // wait_semaphores;
            {}      // signal_semaphores;
        };
        pipelines::volume_renderer::instance().render(render_info);
        
    }


}