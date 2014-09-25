#include <cuda.h>
#include <cudnn.h>

#include <iostream>
#include <mpi.h>
#include <time.h>
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

class cudnnTensor{
private:
	float* data_;
	float* diff_;

	void inline setup(){
		count_ = n_ * c_ * h_ * w_ ;
		data_ = NULL;
		diff_ = NULL;
		CUDNN_CHECK(cudnnCreateTensor4dDescriptor(&descriptor_));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(
			descriptor_,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n_,
			c_,
			h_,
			w_
			));
		CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor_));
		CUDNN_CHECK(cudnnSetFilterDescriptor(
			filter_descriptor_,
			CUDNN_DATA_FLOAT,
			n_,
			c_,
			h_,
			w_
			));
	}

public:
	int n_;
	int c_;
	int h_;
	int w_;

	int count_;

	cudnnTensor4dDescriptor_t descriptor_;
	cudnnFilterDescriptor_t filter_descriptor_;

	cudnnTensor(int n, int c, int h, int w){
		n_ = n;
		c_ = c;
		h_ = h;
		w_ = w;
		setup();
	}

	cudnnTensor(cudnnConvolutionDescriptor_t conv_desc_){
		//Describe output tensor 
		CUDNN_CHECK(cudnnGetOutputTensor4dDim(conv_desc_, CUDNN_CONVOLUTION_FWD,
			&n_, &c_,
			&h_, &w_
			));
		setup();
	}

	inline float* data(){
		if (!data_){
			CUDA_CHECK(cudaMallocManaged(&data_, sizeof(float) * count_));
		}
		return data_;
	}

	inline float* diff(){

		if(!diff_){
			CUDA_CHECK(cudaMallocManaged(&diff_, sizeof(float) * count_));
		}
		return diff_;
	}

	inline cudnnFilterDescriptor_t describe_filter(){
		return filter_descriptor_;
	}

	inline cudnnTensor4dDescriptor_t describe(){
		return descriptor_;
	}

	inline int count(){
		return count_;
	}

};


int main(int argc, char** argv){

	MPI_Init(&argc, &argv);


	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaDeviceReset());

	LOG(DEBUG)<<"cuda started";
	cudnnHandle_t dnn_handle_;
	CUDNN_CHECK(cudnnCreate(&dnn_handle_));
	LOG(DEBUG)<<"cudnn intializaed";
	//TODO: add operations here

	// Create input tensor 

	int n_=128, width_=128, height_=128, channel_=3;
	cudnnTensor* input_blob = new cudnnTensor(n_, channel_, height_, width_);

	float* input_data_ = input_blob->data();
	for (int i=0;i<input_blob->count(); i++){
		input_data_[i] = i;
	}
	LOG(DEBUG)<<"input tensor intialized";

	/*for (int i=0;i<count_input; i++){
	LOG(DEBUG)<<input_data_[i];
	}*/

	// Create filter descriptor
	int k_= 32, c_=channel_, h_=3, w_=3;
	cudnnTensor* filter_blob = new cudnnTensor(k_, c_, h_, w_);

	float* filter_data_ = filter_blob->data();
	for (int i=0;i<filter_blob->count(); i++){
		filter_data_[i] = 1.;
	}
	LOG(DEBUG)<<"filter descriptor intialized";

	// Create conv descriptor
	cudnnConvolutionDescriptor_t conv_desc_;
	CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

	int pad_h_=0,pad_w_=0, u_=1, v_=1, upscalex_=1, upscaley_=1;
	CUDNN_CHECK(cudnnSetConvolutionDescriptor(
		conv_desc_, input_blob->describe(), filter_blob->describe_filter(),
		pad_h_, pad_w_,
		u_, v_,
		upscalex_, upscaley_,
		CUDNN_CONVOLUTION
		));
	LOG(DEBUG)<<"Convolution descriptor initialized";

	//Describe output tensor 
	cudnnTensor* out_blob = new cudnnTensor(conv_desc_);

	LOG(DEBUG)<<"Output tensor initialized";

	double start_conv = MPI_Wtime();

	//Launch convolution
	CUDNN_CHECK(cudnnConvolutionForward(dnn_handle_,
		input_blob->describe(), reinterpret_cast<const void*>(input_data_),
		filter_blob->describe_filter(), reinterpret_cast<const void*>(filter_data_),
		conv_desc_,
		out_blob->describe(), reinterpret_cast<void*>(out_blob->data()),
		CUDNN_RESULT_NO_ACCUMULATE));
	double end_conv = MPI_Wtime();
	LOG(DEBUG)<<"Convolution Done";

	cudaError_t err = cudaDeviceSynchronize();

	float* sum_data_;
	CUDA_CHECK(cudaMallocManaged(&sum_data_, sizeof(float) * filter_blob->count()));
	double start_mpi = MPI_Wtime();
	MPI_Reduce(filter_data_, sum_data_, filter_blob->count(), 
		MPI_FLOAT, MPI_SUM,
		0,
		MPI_COMM_WORLD);

	double end_mpi = MPI_Wtime();

	if (rank ==0){
		LOG(INFO)<<"result:";
		/*float* out_data_ = out_blob->data();
		for (int i= 0; i<out_blob->count();i++){
			cout<<out_data_[i]<<" ";
		}*/

		printf("convolution %g ms taken\n", (end_conv-start_conv)*1000.0);
		printf("mpi %g ms taken\n", (end_mpi-start_mpi)*1000.0);
	}

	//Release resources
	CUDNN_CHECK(cudnnDestroy(dnn_handle_));
	MPI_Finalize();
	return 0;
}