#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fish.h"

__global__ void fishKernel(const FishArray *in_array, FishArray *out_array, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
cudaError_t update_fish_positions_cuda(FishArray *arr, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);

