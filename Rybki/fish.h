#pragma once

const int THREADS_PER_BLOCK = 1024;
const int NUMBER_OF_BLOCKS = 10;
const int NUMBER_OF_FISH = THREADS_PER_BLOCK * NUMBER_OF_BLOCKS;

struct Fish
{
    float x;
    float y;
    float z; // Added z-coordinate
    float vx;
    float vy;
    float vz; // Added z-velocity
};

struct FishArray
{
    float x[NUMBER_OF_FISH];
    float y[NUMBER_OF_FISH];
    float z[NUMBER_OF_FISH]; // Added z-coordinate array
    float vx[NUMBER_OF_FISH];
    float vy[NUMBER_OF_FISH];
    float vz[NUMBER_OF_FISH]; // Added z-velocity array
};

void update_fish_positions_cpu(FishArray *in_array, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
