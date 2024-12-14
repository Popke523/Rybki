#pragma once

struct Fish
{
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct FishArray
{
    float *x;
    float *y;
    float *z;
    float *vx;
    float *vy;
    float *vz;
};

void update_fish_positions_cpu(FishArray *in_array, FishArray *out_array, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
