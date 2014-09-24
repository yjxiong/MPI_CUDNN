#include <cuda.h>
#include <cudnn.h>

#include <iostream>
#include "easylogging++.h"
using namespace std;

//#define CHECK_EQ(a, b, message) \
//do {\
//if (!(a==b)) {\
//	std::cerr << "Assertion `" #a "`==`" #b "` failed in " << __FILE__ \
//	<< " line " << __LINE__ << ": " << message << std::endl; \
//	std::exit(EXIT_FAILURE); \
//} \
//} while (false)

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
do {\
	cudaError_t error = condition; \
	CHECK_EQ(error, cudaSuccess)<< cudaGetErrorString(error); \
} while (0)

// CUDNN: various checks for different function calls.
#define CUDNN_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
do {\
	cudnnStatus_t error = condition; \
	CHECK_EQ(error, CUDNN_STATUS_SUCCESS)<< "CUDNN Failed"; \
} while (0)


_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv){
	CUDA_CHECK(cudaSetDevice(0));
	LOG(DEBUG)<<"cuda started";
	cudnnHandle_t dnn_handle_;
	CUDNN_CHECK(cudnnCreate(&dnn_handle_));
	LOG(DEBUG)<<"cudnn intializaed";
	//TODO: add operations here

	// Create input tensor 
	cudnnTensor4dDescriptor_t input_tensor_;
	CUDNN_CHECK(cudnnCreateTensor4dDescriptor(&input_tensor_));

	int n_=5, width_=5, height_=5, channel_=5;
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		input_tensor_,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		channel_,
		n_,
		width_,
		height_
		));
	float* input_data_;
	float count_input =  n_ * width_ * height_ * channel_;
	CUDA_CHECK(cudaMallocManaged(&input_data_, 
		sizeof(float) * count_input));

	for (int i=0;i<count_input; i++){
		input_data_[i] = 1;
	}
	LOG(DEBUG)<<"input tensor intialized";

	/*for (int i=0;i<count_input; i++){
		LOG(DEBUG)<<input_data_[i];
	}*/

	// Create filter descriptor
	int k_= 5, c_=channel_, h_=3, w_=3;
	cudnnFilterDescriptor_t filter_desc_;
	CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));

	CUDNN_CHECK(cudnnSetFilterDescriptor(filter_desc_,
		CUDNN_DATA_FLOAT,
		k_,
		c_,
		h_,
		w_));

	float* filter_data_;

	//allocate filter data
	float count_filter = k_ * c_ * h_ * w_;

	CUDA_CHECK(cudaMallocManaged(&filter_data_, 
		sizeof(float) * count_filter));
	for (int i=0;i<count_filter; i++){
		filter_data_[i] = 1.;
	}
	LOG(DEBUG)<<"filter descriptor intialized";

	// Create conv descriptor
	cudnnConvolutionDescriptor_t conv_desc_;
	CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

	int pad_h_=0,pad_w_=0, u_=1, v_=1, upscalex_=1, upscaley_=1;
	CUDNN_CHECK(cudnnSetConvolutionDescriptor(
		conv_desc_, input_tensor_, filter_desc_,
		pad_h_, pad_w_,
		u_, v_,
		upscalex_, upscaley_,
		CUDNN_CONVOLUTION
	));
	LOG(DEBUG)<<"Convolution descriptor initialized";

	int out_n_, out_c_, out_h_, out_w_;


	//Describe output tensor 
	CUDNN_CHECK(cudnnGetOutputTensor4dDim(conv_desc_, CUDNN_CONVOLUTION_FWD,
		&out_n_, &out_c_,
		&out_h_, &out_w_
		));

	int count_out = out_n_ * out_c_ * out_h_ * out_w_;
	cudnnTensor4dDescriptor_t out_tensor_;
	CUDNN_CHECK(cudnnCreateTensor4dDescriptor(&out_tensor_));

	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		out_tensor_,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		out_c_,
		out_n_,
		out_h_,
		out_w_
		));

	float* out_data_;
	CUDA_CHECK(cudaMallocManaged(&out_data_, 
		sizeof(float) * count_out));
	for (int i=0;i<count_out; i++){
		out_data_[i] = 0.;
	}

	LOG(DEBUG)<<"Output tensor initialized";

	//Launch convolution
	CUDNN_CHECK(cudnnConvolutionForward(dnn_handle_,
		input_tensor_, reinterpret_cast<const void*>(input_data_),
		filter_desc_, reinterpret_cast<const void*>(filter_data_),
		conv_desc_,
		out_tensor_, reinterpret_cast<void*>(out_data_),
		CUDNN_RESULT_ACCUMULATE));

	LOG(DEBUG)<<"Convolution Done";

	cudaError_t err = cudaDeviceSynchronize();
	LOG(INFO)<<"result:";
	for (int i=0;i<count_out; i++){
		LOG(INFO)<<out_data_[i];
	}

	//Release resources
	CUDNN_CHECK(cudnnDestroy(dnn_handle_));
	
	return 0;
}