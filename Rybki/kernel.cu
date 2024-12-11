#include "kernel.h"
#include <cstdio>
#include <cmath>

__constant__ const float SPEED_FACTOR = 0.1f;
__constant__ const float MARGIN = 0.1f;

// CUDA Kernel
__global__ void fishKernel(const FishArray *in_array, FishArray *out_array, int number_of_fish, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
    int fish_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fish_idx >= number_of_fish) return;

    Fish fish = {
        in_array->x[fish_idx],
        in_array->y[fish_idx],
        in_array->z[fish_idx],
        in_array->vx[fish_idx],
        in_array->vy[fish_idx],
        in_array->vz[fish_idx]
    };

    Fish otherFish;

    float dx, dy, dz;
    float distance_squared;

    float xpos_avg = 0, ypos_avg = 0, zpos_avg = 0;
    float xvel_avg = 0, yvel_avg = 0, zvel_avg = 0;
    float neighbors = 0;
    float close_dx = 0, close_dy = 0, close_dz = 0;
    float speed;

    for (int i = 0; i < number_of_fish; i++)
    {
        if (i == fish_idx) continue;

        otherFish.x = in_array->x[i];
        otherFish.y = in_array->y[i];
        otherFish.z = in_array->z[i];
        otherFish.vx = in_array->vx[i];
        otherFish.vy = in_array->vy[i];
        otherFish.vz = in_array->vz[i];

        dx = fish.x - otherFish.x;
        dy = fish.y - otherFish.y;
        dz = fish.z - otherFish.z;
        distance_squared = dx * dx + dy * dy + dz * dz;

        // Separation - steer away from another fish in protected range 
        if (distance_squared < protected_range * protected_range)
        {
            close_dx += dx;
            close_dy += dy;
            close_dz += dz;
        }

		// Alignment and Cohesion - steer towards the average position and velocity of other fish in visible range
        if (distance_squared < visible_range * visible_range)
        {
            xpos_avg += otherFish.x;
            ypos_avg += otherFish.y;
            zpos_avg += otherFish.z;
            xvel_avg += otherFish.vx;
            yvel_avg += otherFish.vy;
            zvel_avg += otherFish.vz;
            neighbors++;
        }
    }

	// Alignment and Cohesion - update 
    if (neighbors > 0)
    {
        xpos_avg /= neighbors;
        ypos_avg /= neighbors;
        zpos_avg /= neighbors;
        xvel_avg /= neighbors;
        yvel_avg /= neighbors;
        zvel_avg /= neighbors;

        fish.vx += (xpos_avg - fish.x) * centering_factor + (xvel_avg - fish.vx) * matching_factor;
        fish.vy += (ypos_avg - fish.y) * centering_factor + (yvel_avg - fish.vy) * matching_factor;
        fish.vz += (zpos_avg - fish.z) * centering_factor + (zvel_avg - fish.vz) * matching_factor;
    }

    fish.vx += close_dx * avoid_factor;
    fish.vy += close_dy * avoid_factor;
    fish.vz += close_dz * avoid_factor;

	// turn the fish if it is too close to the edge
    if (fish.x < MARGIN - 1)
    {
        fish.vx += turn_factor;
    }
    else if (fish.x > 1 - MARGIN)
    {
        fish.vx -= turn_factor;
    }
    if (fish.y < MARGIN - 1)
    {
        fish.vy += turn_factor;
    }
    else if (fish.y > 1 - MARGIN)
    {
        fish.vy -= turn_factor;
    }
    if (fish.z < MARGIN - 1)
    {
        fish.vz += turn_factor;
    }
    else if (fish.z > 1 - MARGIN)
    {
        fish.vz -= turn_factor;
    }

	// limit the speed of the fish
    speed = sqrtf(fish.vx * fish.vx + fish.vy * fish.vy + fish.vz * fish.vz);
    if (speed < min_speed)
    {
        fish.vx = (fish.vx / speed) * min_speed;
        fish.vy = (fish.vy / speed) * min_speed;
        fish.vz = (fish.vz / speed) * min_speed;
    }
    else if (speed > max_speed)
    {
        fish.vx = (fish.vx / speed) * max_speed;
        fish.vy = (fish.vy / speed) * max_speed;
        fish.vz = (fish.vz / speed) * max_speed;
    }

    fish.x += fish.vx * SPEED_FACTOR;
    fish.y += fish.vy * SPEED_FACTOR;
    fish.z += fish.vz * SPEED_FACTOR;

    out_array->x[fish_idx] = fish.x;
    out_array->y[fish_idx] = fish.y;
    out_array->z[fish_idx] = fish.z;
    out_array->vx[fish_idx] = fish.vx;
    out_array->vy[fish_idx] = fish.vy;
    out_array->vz[fish_idx] = fish.vz;
}

// Helper function for using CUDA to update fish positions in parallel.
cudaError_t update_fish_positions_cuda(FishArray *arr, FishArray *dev_old, FishArray *dev_new, int number_of_fish, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{

    cudaError_t cudaStatus;

	// calculate the number of blocks and threads
    int THREADS_PER_BLOCK = number_of_fish > 512 ? 512 : number_of_fish;
	int NUMBER_OF_BLOCKS = (number_of_fish + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel with updated parameters.
    fishKernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (dev_old, dev_new, number_of_fish, visible_range, protected_range, avoid_factor, matching_factor, centering_factor, turn_factor, min_speed, max_speed);

    // Check for any errors launching the kernel.
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "fishKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fishKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output array from GPU buffer to host memory.
	cudaStatus = copy_fish_array_device_to_host(arr, dev_new, number_of_fish);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    std::swap(dev_old, dev_new);

    return cudaStatus;

Error:

    free_fish_array(dev_old);
	free_fish_array(dev_new);

    return cudaStatus;
}

cudaError_t allocate_fish_array(FishArray *arr, int number_of_fish)
{
    cudaError_t cudaStatus;

    // Allocate GPU buffers for input and output arrays.
	cudaStatus = cudaMalloc((void **)&arr->x, number_of_fish * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void **)&arr->y, number_of_fish * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&arr->z, number_of_fish * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&arr->vx, number_of_fish * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&arr->vy, number_of_fish * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&arr->vz, number_of_fish * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    return cudaStatus;

Error:
	cudaFree(arr->x);
	cudaFree(arr->y);
	cudaFree(arr->z);
	cudaFree(arr->vx);
	cudaFree(arr->vy);
	cudaFree(arr->vz);
    
	return cudaStatus;
}


cudaError_t free_fish_array(FishArray *arr)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(arr->x);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }
    cudaStatus = cudaFree(arr->y);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }
    cudaStatus = cudaFree(arr->z);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }
    cudaStatus = cudaFree(arr->vx);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }
    cudaStatus = cudaFree(arr->vy);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }
    cudaStatus = cudaFree(arr->vz);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed!");
        goto Error;
    }

    return cudaStatus;

Error:
    cudaFree(arr->x);
    cudaFree(arr->y);
    cudaFree(arr->z);
    cudaFree(arr->vx);
    cudaFree(arr->vy);
    cudaFree(arr->vz);

    return cudaStatus;
}

cudaError_t initial_copy(FishArray *dev_arr, FishArray arr, int number_of_fish)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(dev_arr->x, arr.x, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr->y, arr.y, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr->z, arr.z, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr->vx, arr.vx, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr->vy, arr.vy, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr->vz, arr.vz, number_of_fish * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    return cudaStatus;

Error:
	free_fish_array(dev_arr);

    return cudaStatus;
}


cudaError_t copy_fish_array_device_to_host(FishArray *arr, FishArray *dev_arr, int number_of_fish)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(arr->x, dev_arr->x, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(arr->y, dev_arr->y, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(arr->z, dev_arr->z, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(arr->vx, dev_arr->vx, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(arr->vy, dev_arr->vy, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(arr->vz, dev_arr->vz, number_of_fish * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    return cudaStatus;

Error:
    free_fish_array(dev_arr);
	return cudaStatus;
}