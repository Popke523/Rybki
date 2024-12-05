#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "shader_s/shader_s.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// Shader source code
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in mat3 instanceMatrix;

uniform mat3 projection;

void main() {
    vec3 pos = instanceMatrix * vec3(aPos, 1.0);
    gl_Position = vec4(projection * pos, 1.0);
}
)";

const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0); // Orange color
}
)";

// Function to compile shaders
GLuint compileShader(GLenum type, const char *source)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);

	int success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 512, NULL, infoLog);
		std::cerr << "Shader compilation failed: " << infoLog << std::endl;
	}
	return shader;
}

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

const int THREADS_PER_BLOCK = 128;
const int NUMBER_OF_BLOCKS = 1;
const int NUMBER_OF_FISH = THREADS_PER_BLOCK * NUMBER_OF_BLOCKS;

#define SPEED_FACTOR 0.1f
__constant__ const float MARGIN = 0.1f;

struct Fish
{
	float x;
	float y;
	float vx;
	float vy;
};

struct FishArray
{
	float x[NUMBER_OF_FISH];
	float y[NUMBER_OF_FISH];
	float vx[NUMBER_OF_FISH];
	float vy[NUMBER_OF_FISH];
};

cudaError_t update_fish_positions(FishArray *arr, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

__global__ void fishKernel(const FishArray *in_array, FishArray *out_array, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
	int fish_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (fish_idx >= size) return;

	Fish fish = {
		in_array->x[fish_idx],
		in_array->y[fish_idx],
		in_array->vx[fish_idx],
		in_array->vy[fish_idx]
	};

	Fish otherFish;

	float dx;
	float dy;
	float distance_squared;

	float xpos_avg = 0;
	float ypos_avg = 0;
	float xvel_avg = 0;
	float yvel_avg = 0;
	float neighbors = 0;
	float close_dx = 0;
	float close_dy = 0;
	float speed;

	for (int i = 0; i < size; i++)
	{
		if (i == fish_idx) continue;

		otherFish.x = in_array->x[i];
		otherFish.y = in_array->y[i];
		otherFish.vx = in_array->vx[i];
		otherFish.vy = in_array->vy[i];

		dx = fish.x - otherFish.x;
		dy = fish.y - otherFish.y;
		distance_squared = dx * dx + dy * dy;

		if (distance_squared < protected_range * protected_range)
		{
			close_dx = fish.x - otherFish.x;
			close_dy = fish.y - otherFish.y;
		}

		if (distance_squared < visible_range * visible_range)
		{
			xpos_avg += otherFish.x;
			ypos_avg += otherFish.y;
			xvel_avg += otherFish.vx;
			yvel_avg += otherFish.vy;
			neighbors++;
		}
	}

	if (neighbors > 0)
	{
		xpos_avg /= neighbors;
		ypos_avg /= neighbors;
		xvel_avg /= neighbors;
		yvel_avg /= neighbors;

		fish.vx += (xpos_avg - fish.x) * centering_factor + (xvel_avg - fish.vx) * matching_factor;
		fish.vy += (ypos_avg - fish.y) * centering_factor + (yvel_avg - fish.vy) * matching_factor;
	}

	fish.vx += close_dx * avoid_factor;
	fish.vy += close_dy * avoid_factor;

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


	speed = sqrt(fish.vx * fish.vx + fish.vy * fish.vy);
	if (speed < min_speed)
	{
		fish.vx = fish.vx / speed * min_speed;
		fish.vy = fish.vy / speed * min_speed;
	}
	else if (speed > max_speed)
	{
		fish.vx = fish.vx / speed * max_speed;
		fish.vy = fish.vy / speed * max_speed;
	}

	fish.x += fish.vx * SPEED_FACTOR;
	fish.y += fish.vy * SPEED_FACTOR;

	out_array->x[fish_idx] = fish.x;
	out_array->y[fish_idx] = fish.y;
	out_array->vx[fish_idx] = fish.vx;
	out_array->vy[fish_idx] = fish.vy;
}

int main()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	(void)io;

	// Set up ImGui backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	// Set up ImGui style
	ImGui::StyleColorsDark();

	// Variables for sliders
	float visible_range = 0.1f;
	float protected_range = 0.02f;
	float avoid_factor = 0.2f;
	float matching_factor = 0.02f;
	float centering_factor = 0.0005f;
	float turn_factor = 0.001f;
	float min_speed = 0.02f;
	float max_speed = 0.03f;

	// Compile shaders and link the shader program
	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);


	// initialize the array with random positions and velocities
	FishArray arr;
	std::srand(std::time(nullptr));
	for (int i = 0; i < NUMBER_OF_FISH; i++)
	{
		arr.x[i] = (float)std::rand() / RAND_MAX - 0.5f;
		arr.y[i] = (float)std::rand() / RAND_MAX - 0.5f;
		arr.vx[i] = ((float)std::rand() / RAND_MAX - 0.5f) * SPEED_FACTOR;
		arr.vy[i] = ((float)std::rand() / RAND_MAX - 0.5f) * SPEED_FACTOR;
	}

	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float triangleVertices[] = {
		// positions         
		  -0.01f,0.0f,0.0f,
		  -0.02f,-0.02f,-0.01f,
		  -0.02f,-0.02f,0.01f,
		  0.04f,0.0f,0.0f,
		  -0.02f,0.02f,-0.01f,
		  -0.02f,0.02f,0.01f
	};

	glm::mat3 instanceMatrices[NUMBER_OF_FISH] = { glm::mat3(1.0f) };

	// Create buffers and upload data
	GLuint VAO, VBO, instanceVBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &instanceVBO);

	// Set up triangle vertex data
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	// glBindVertexArray(0);

	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		// render
		// ------
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Use the shader program
		glUseProgram(shaderProgram);

		// Add vectors in parallel.
		cudaStatus = update_fish_positions(&arr, NUMBER_OF_FISH, visible_range, protected_range, avoid_factor, matching_factor, centering_factor, turn_factor, min_speed, max_speed);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		for (int i = 0; i < NUMBER_OF_FISH; i++)
		{
			float angle = atan2f(arr.vy[i], arr.vx[i]);
			glm::mat3 transform = glm::mat3(1.0f);
			transform[0][0] = cos(angle);
			transform[0][1] = sin(angle);
			transform[1][0] = -sin(angle);
			transform[1][1] = cos(angle);
			transform[2][0] = arr.x[i];
			transform[2][1] = arr.y[i];

			instanceMatrices[i] = transform;
		}

		glm::mat3 projection = glm::mat3(1.0f);

		// Set up instance matrix data
		glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, NUMBER_OF_FISH * sizeof(glm::mat3), &instanceMatrices[0], GL_STATIC_DRAW);
		for (int i = 0; i < 3; ++i)
		{
			glEnableVertexAttribArray(1 + i);
			glVertexAttribPointer(1 + i, 3, GL_FLOAT, GL_FALSE, sizeof(glm::mat3), (void *)(sizeof(glm::vec3) * i));
			glVertexAttribDivisor(1 + i, 1); // Update once per instance
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		// Upload projection matrix
		GLint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
		glUniformMatrix3fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

		// Draw instances
		glBindVertexArray(VAO);
		glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, NUMBER_OF_FISH);

		// Start new ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Create GUI
		ImGui::Begin("Simple GUI");
		ImGui::Text("Adjust the sliders:");
		ImGui::SliderFloat("Visible Range", &visible_range, 0.0f, 1.0f);
		ImGui::SliderFloat("Protected Range", &protected_range, 0.0f, 0.1f);
		ImGui::SliderFloat("Avoid Factor", &avoid_factor, 0.0f, 1.0f);
		ImGui::SliderFloat("Matching Factor", &matching_factor, 0.0f, 0.1f);
		ImGui::SliderFloat("Centering Factor", &centering_factor, 0.0f, 0.1f);
		ImGui::SliderFloat("Turn Factor", &turn_factor, 0.0f, 0.1f);
		ImGui::SliderFloat("Min Speed", &min_speed, 0.0f, 1.0f);
		ImGui::SliderFloat("Max Speed", &max_speed, 0.0f, 1.0f);
		if (ImGui::Button("Reset"))
		{
			visible_range = 0.1f;
			protected_range = 0.02f;
			avoid_factor = 0.2f;
			matching_factor = 0.02f;
			centering_factor = 0.0005f;
			turn_factor = 0.001f;
			min_speed = 0.02f;
			max_speed = 0.03f;
		}

		ImGui::Text("Frame rate: %.1f FPS", ImGui::GetIO().Framerate);
		ImGui::End();

		// Render ImGui
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());




		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);

	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t update_fish_positions(FishArray *arr, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
	FishArray *dev_a = 0;
	FishArray *dev_b = 0;

	cudaError_t cudaStatus;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void **)&dev_a, sizeof(FishArray));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, sizeof(FishArray));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, arr, sizeof(FishArray), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	fishKernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (dev_a, dev_b, NUMBER_OF_FISH, visible_range, protected_range, avoid_factor, matching_factor, centering_factor, turn_factor, min_speed, max_speed);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arr, dev_b, sizeof(FishArray), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
	float aspect_ratio = 1.0f; // Desired aspect ratio
	int viewport_width, viewport_height;

	if (width / (float)height > aspect_ratio)
	{
		viewport_height = height;
		viewport_width = (int)(height * aspect_ratio);
	}
	else
	{
		viewport_width = width;
		viewport_height = (int)(width / aspect_ratio);
	}

	int viewport_x = (width - viewport_width) / 2;
	int viewport_y = (height - viewport_height) / 2;

	glViewport(viewport_x, viewport_y, viewport_width, viewport_height);
}