#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fish.h"

__global__ void fishKernel(float *in_x, float *in_y, float *in_z, float *in_vx, float *in_vy, float *in_vz,
	float *out_x, float *out_y, float *out_z, float *out_vx, float *out_vy, float *out_vz,
	int number_of_fish, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
cudaError_t update_fish_positions_cuda(FishArray *arr, FishArray *dev_old, FishArray *dev_new, int number_of_fish, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
cudaError_t allocate_fish_array(FishArray *arr, int number_of_fish);
cudaError_t free_fish_array(FishArray *arr);
cudaError_t initial_copy(FishArray *dev_arr, FishArray arr, int number_of_fish);
cudaError_t copy_fish_array_device_to_host(FishArray *arr, FishArray *dev_arr, int number_of_fish);
