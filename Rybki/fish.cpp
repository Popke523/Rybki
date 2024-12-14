#include "fish.h"
#include <cmath>
#include <cstring>
#include <algorithm>

const float SPEED_FACTOR = 0.1f;
const float MARGIN = 0.1f;

void update_fish_positions_cpu(FishArray *in_array, FishArray *out_array, int number_of_fish, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
	for (int fish_idx = 0; fish_idx < number_of_fish; fish_idx++)
	{
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

			if (distance_squared < protected_range * protected_range)
			{
				close_dx += dx;
				close_dy += dy;
				close_dz += dz;
			}

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

	std::swap(in_array->x, out_array->x);
	std::swap(in_array->y, out_array->y);
	std::swap(in_array->z, out_array->z);
	std::swap(in_array->vx, out_array->vx);
	std::swap(in_array->vy, out_array->vy);
	std::swap(in_array->vz, out_array->vz);
}
