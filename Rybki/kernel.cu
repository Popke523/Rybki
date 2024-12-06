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

// Shader source code with lighting
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in mat4 instanceMatrix;

uniform mat4 projection;
uniform mat4 view;

out vec3 FragPos;

void main() {
    vec4 worldPos = instanceMatrix * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);
    gl_Position = projection * view * worldPos;
}
)";

const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;

// Light properties
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(vec3(0.0, 0.0, 1.0)); // Simple normal for pyramids
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
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

// Camera parameters for orbiting around the origin
float radius = 5.0f;
float yaw = -90.0f;
float pitch = 0.0f;
bool firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool leftMousePressed = false;

// Derived camera position
glm::vec3 cameraPos;
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

cudaError_t update_fish_positions(FishArray *arr, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);

// CUDA Kernel
__global__ void fishKernel(const FishArray *in_array, FishArray *out_array, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
    int fish_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fish_idx >= size) return;

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

    for (int i = 0; i < size; i++)
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
            close_dx = dx;
            close_dy = dy;
            close_dz = dz;
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
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL 3D Pyramid", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

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
        arr.z[i] = (float)std::rand() / RAND_MAX - 0.5f;
        arr.vx[i] = ((float)std::rand() / RAND_MAX - 0.5f) * SPEED_FACTOR;
        arr.vy[i] = ((float)std::rand() / RAND_MAX - 0.5f) * SPEED_FACTOR;
        arr.vz[i] = ((float)std::rand() / RAND_MAX - 0.5f) * SPEED_FACTOR;
    }

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float pyramidVertices[] = {
        // Base vertices
        -0.02f, -0.02f, 0.0f,
         0.02f, -0.02f, 0.0f,
         0.02f,  0.02f, 0.0f,
        -0.02f,  0.02f, 0.0f,
        // Apex vertex
         0.0f,   0.0f,  0.04f
    };

    // Define indices for the pyramid (base and sides)
    unsigned int indices[] = {
        // Base
        0, 1, 2,
        0, 2, 3,
        // Sides
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4
    };

    glm::mat4 instanceMatrices[NUMBER_OF_FISH];
    for (int i = 0; i < NUMBER_OF_FISH; ++i)
    {
        instanceMatrices[i] = glm::mat4(1.0f);
    }

    // Create buffers and upload data
    GLuint VAO, VBO, EBO, instanceVBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &instanceVBO);

    // Set up pyramid vertex data
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Set up element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Set up instance matrix data
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, NUMBER_OF_FISH * sizeof(glm::mat4), &instanceMatrices[0], GL_DYNAMIC_DRAW);
    std::size_t vec4Size = sizeof(glm::vec4);
    for (int i = 0; i < 4; ++i)
    {
        glEnableVertexAttribArray(1 + i);
        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(vec4Size * i));
        glVertexAttribDivisor(1 + i, 1); // Update once per instance
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Enable depth testing for 3D
    glEnable(GL_DEPTH_TEST);

    // Initialize camera position based on initial angles
    cameraPos.x = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraPos.y = radius * sin(glm::radians(pitch));
    cameraPos.z = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    // Light properties
    glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor(1.0f, 0.5f, 0.2f); // Orange color

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear depth buffer

        // Update camera position based on yaw and pitch
        cameraPos.x = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraPos.y = radius * sin(glm::radians(pitch));
        cameraPos.z = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));

        // Use the shader program
        glUseProgram(shaderProgram);

        // Set light and view position uniforms
        GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
        GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
        GLint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
        GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(cameraPos));
        glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
        glUniform3fv(objectColorLoc, 1, glm::value_ptr(objectColor));

        // Update view and projection matrices
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Update fish positions using CUDA
        cudaStatus = update_fish_positions(&arr, NUMBER_OF_FISH, visible_range, protected_range, avoid_factor, matching_factor, centering_factor, turn_factor, min_speed, max_speed);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "update_fish_positions failed!");
            return 1;
        }

        // Update instance matrices based on new positions and velocities
        for (int i = 0; i < NUMBER_OF_FISH; i++)
        {
            float angle_yaw = atan2f(arr.vy[i], arr.vx[i]);
            float angle_pitch = atan2f(arr.vz[i], sqrtf(arr.vx[i] * arr.vx[i] + arr.vy[i] * arr.vy[i]));

            glm::mat4 transform = glm::mat4(1.0f);
            transform = glm::translate(transform, glm::vec3(arr.x[i], arr.y[i], arr.z[i]));
            transform = glm::rotate(transform, angle_yaw, glm::vec3(0.0f, 1.0f, 0.0f));   // Yaw
            transform = glm::rotate(transform, angle_pitch, glm::vec3(1.0f, 0.0f, 0.0f)); // Pitch

            instanceMatrices[i] = transform;
        }

        // Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUMBER_OF_FISH * sizeof(glm::mat4), &instanceMatrices[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Draw instances
        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0, NUMBER_OF_FISH);
        glBindVertexArray(0);

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
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &instanceVBO);

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

// Helper function for using CUDA to update fish positions in parallel.
cudaError_t update_fish_positions(FishArray *arr, int size, float visible_range, float protected_range, float avoid_factor, float matching_factor, float centering_factor, float turn_factor, float min_speed, float max_speed)
{
    FishArray *dev_a = 0;
    FishArray *dev_b = 0;

    cudaError_t cudaStatus;
    // Allocate GPU buffers for input and output arrays.
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

    // Copy input array from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, arr, sizeof(FishArray), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch the kernel with updated parameters.
    fishKernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (dev_a, dev_b, NUMBER_OF_FISH, visible_range, protected_range, avoid_factor, matching_factor, centering_factor, turn_factor, min_speed, max_speed);

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

// Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
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

    if ((float)width / (float)height > aspect_ratio)
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

// glfw: mouse button callback to handle left click
// -------------------------------------------------
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            leftMousePressed = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else if (action == GLFW_RELEASE)
        {
            leftMousePressed = false;
            firstMouse = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

// glfw: cursor position callback to handle camera movement
// --------------------------------------------------------
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!leftMousePressed)
        return;

    if (firstMouse)
    {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos) - lastX;
    float yoffset = lastY - static_cast<float>(ypos); // reversed since y-coordinates go from bottom to top
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);

    float sensitivity = 0.1f; // change this value to your liking
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;
}
