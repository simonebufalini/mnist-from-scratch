#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
	size_t input_sz;	// 784
	size_t hidden_sz;	// 128
	size_t output_sz;	// 10

	float *h_layer;		// buffer for hidden layer 
	float *o_layer;		// buffer for outpu layer
						// input is in the dataset and will not be copied here

	float *delta_h;		// reusable buffer for gradients (avoid malloc/free inside training loop)
	float *delta_o;

	float *w_ih;		// weights input-hidden layer
	float *w_ho;		// weights hidden-output layer
	float *b_h;			// bias hidden layer
	float *b_o;			// bias output layer
} NN;

static inline float relu(float x) {
	return x > 0 ? x : 0;
}

static inline float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

static inline float relu_deriv(float x) {
	return x > 0 ? 1.0f : 0.0f;
}

static inline float sigmoid_deriv(float x) {
	return x * (1.0f - x);
}

static inline float rand_weight(void) {
	return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

NN *nn_init(size_t i, size_t h, size_t o);
void nn_free(NN *nn);

void nn_forward(NN *nn, const float *input);

// executes a single iteration of the training mechanism (updates weights, batch size = 1)
float nn_train(NN *nn, const float *input, int target_lbl, float lr);

#endif
