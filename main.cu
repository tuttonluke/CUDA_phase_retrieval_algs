#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#include <stdio.h>

#include <iostream>

// Kernel for magnitude enforcement (restriction on the SLM)
__global__ void phase_constraint(cufftComplex* data, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		cufftComplex value = data[idx];
		float phase = atan2f(value.y, value.x); // get phase
		value.x = cosf(phase); // update real 
		value.y = sinf(phase); // update imaginary
		data[idx] = value;
	}
}

// Kernel for restriction on the image
__global__ void image_restriction(cufftComplex* data, float* amplitude, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		cufftComplex value = data[idx];
		float current_magnitude = hypotf(value.x, value.y);
		value.x /= (current_magnitude*amplitude[idx]);
		value.y /= (current_magnitude * amplitude[idx]);
		data[idx] = value;
	}
}

int main()
{

  // Load in image
  int img_width{};
  int img_height{};
  int img_channels{};
  const char * img_file_path = "C:\\Users\\luket\\Documents\\C++_learning\\CUDA_learning\\test_project\\butterfly.jpg";
  unsigned char *img = stbi_load(img_file_path, &img_width, &img_height, &img_channels, 0);

  if (img == NULL) {
      std::cout << "Error loading image" << std::endl;
      exit(-1);
  }
  std::cout << "Loaded image with witdth: " << img_width << ", height: " << img_height << ", and " << img_channels << " channels." << std::endl;
	
  // 
  const unsigned int N{ 1024 };
	const unsigned int size = N * N;
	const float pi = 3.14159f;

	// Allocate host memory
	std::vector<cufftComplex> h_input(sizeof(cufftComplex) * size);
	std::vector<float> h_amplitude(sizeof(float) * size); // amplitude constraint

	// Initialise input and magnitude
	srand(time(0)); // seed random number generator
	for (int i = 0; i < size; ++i) {
		float random_phase = static_cast<float>(rand()) / RAND_MAX * 2 * pi; // Random phase between 0 and 2π
		h_input[i].x = cosf(random_phase); // real part
		h_input[i].y = sinf(random_phase); // imaginary part
		h_amplitude[i] = 1.0f; // TODO (load image)
	}

	// Allocate device memory
	cufftComplex* d_data{};
	float* d_amplitude{};
	cudaMalloc(&d_data, sizeof(cufftComplex) * size);
	cudaMalloc(&d_amplitude, sizeof(float) * size);

	// Copy to device
	cudaMemcpy(d_data, h_input.data(), sizeof(cufftComplex) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_amplitude, h_amplitude.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

	// Create FFT plan
	cufftHandle plan{ 0 };
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	// Number of iterations
	const int iterations{ 100 };
	
	dim3 block_size{ 1024 };
	dim3 grid_size{ size / block_size.x };

	// GS algorith loop
	for (int iter = 0; iter < iterations; ++iter) {
		// Forward FFT
		cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

		// Enforce magnitude constraint in spatial domain
		image_restriction << <grid_size, block_size >> > (d_data, d_amplitude, size);

		// Inverse FFT
		cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

		// Enforce magnitude constraint in frequency domain
		phase_constraint <<<grid_size, block_size>>>(d_data, size);
	}
	
	// Copy results back to host
	cudaMemcpy(h_input.data(), d_data, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
	
	// Free GPU memory
	cudaFree(d_data);
	cudaFree(d_amplitude);

  // Write image
  stbi_write_jpg("output.jpg", img_width, img_height, 3, img, 100);
  stbi_image_free(img);

	return 0;
}