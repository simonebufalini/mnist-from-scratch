#include "nn.h"
#include <stddef.h>

NN *nn_init(size_t i, size_t h, size_t o) {
	NN *nn = malloc(sizeof(NN));
	// assign the nn parameters	
	nn->input_sz = i;
	nn->hidden_sz = h;
	nn->output_sz = o;
	
	// allocating buffer for neurons and weights
	nn->h_layer = malloc(h * sizeof(float));
	nn->o_layer = malloc(o * sizeof(float));

	nn->delta_h = malloc(h * sizeof(float));
	nn->delta_o = malloc(o * sizeof(float));

	nn->w_ih = malloc(i * h * sizeof(float));
	nn->w_ho = malloc(h * o * sizeof(float));
	nn->b_h = malloc(h * sizeof(float));
	nn->b_o = malloc(o * sizeof(float));

	// initializing with small random values
	// hidden -> output
	for (int k = 0; k < h*o; k++)
		nn->w_ho[k] = rand_weight();
	
	// input -> hidden
	for (int k = 0; k < i*h; k++)
		nn->w_ih[k] = rand_weight();

	// bias
	for (int k = 0; k < h; k++) nn->b_h[k] = rand_weight();
	for (int k = 0; k < o; k++) nn->b_o[k] = rand_weight();

	return nn;
}

void nn_free(NN *nn) {
	free(nn->h_layer);
	free(nn->o_layer);

	free(nn->delta_h);
	free(nn->delta_o);

	free(nn->w_ih);
	free(nn->w_ho);
	free(nn->b_h);
	free(nn->b_o);

	free(nn);
}

// FORWARD PASS.
// inversion of logic: pushing the value to the next level
// neurons instead of pulling from behind to access memory
// sequentially for better cache usage.

void nn_forward(NN *nn, const float *input) {
	// initialize neurons with the bias for input->hidden
	for (size_t j = 0; j < nn->hidden_sz; j++)
		nn->h_layer[j] = nn->b_h[j];

	for (size_t i = 0; i < nn->input_sz; i++) {
		
		float in = input[i];

		for (size_t j = 0; j < nn->hidden_sz; j++)
			nn->h_layer[j] += in * nn->w_ih[i * nn->hidden_sz + j];
	}
	
	// apply relu
	for (size_t j = 0; j < nn->hidden_sz; j++)
		nn->h_layer[j] = relu(nn->h_layer[j]);

	// processing hidden->output with the same inversin of logic
	// initialize with the bias for hidden->output
	for (size_t k = 0; k < nn->output_sz; k++) 
		nn->o_layer[k] = nn->b_o[k];

	for (size_t j = 0; j < nn->hidden_sz; j++) {
		
		float hid = nn->h_layer[j];

		for (size_t k = 0; k < nn->output_sz; k++)
			nn->o_layer[k] += hid * nn->w_ho[j * nn->output_sz + k];
	}

	// apply sigmoid
	for (size_t k = 0; k < nn->output_sz; k++)
		nn->o_layer[k] = sigmoid(nn->o_layer[k]);
}

float nn_train(NN *nn, const float *input, int target_lbl, float lr) {
	// first calculate the output
	nn_forward(nn, input);

	float loss = 0.0f;
	
	// calculating output delta
	for (size_t k = 0; k < nn->output_sz; k++) {
		float tgt = (k == target_lbl) ? 1.0f : 0.0f;
		float err = tgt - nn->o_layer[k];
		
		// loss (MSE)
		loss += err*err;

		nn->delta_o[k] = err * sigmoid_deriv(nn->o_layer[k]);
	}

	// calculating hidden delta
	for (size_t j = 0; j < nn->hidden_sz; j++) {
		float err_sum = 0.0f;

		for (size_t k = 0; k < nn->output_sz; k++)
			err_sum += nn->delta_o[k] * nn->w_ho[j * nn->output_sz + k];

		nn->delta_h[j] = err_sum * relu_deriv(nn->h_layer[j]);
	}

	// GRADIENT DESCENT
	// weights hidden->output
	for (size_t j = 0; j < nn->hidden_sz; j++) {
		float h_val = nn->h_layer[j];

		for (size_t k = 0; k < nn->output_sz; k++)
			nn->w_ho[j * nn->output_sz + k] += nn->delta_o[k] * h_val * lr;
	}

	// bias output
	for (size_t k = 0; k < nn->output_sz; k++)
		nn->b_o[k] += nn->delta_o[k] * lr;

	// weights input->hidden
	for (size_t i = 0; i < nn->input_sz; i++) {
		float in = input[i];

		for (size_t j = 0; j < nn->hidden_sz; j++)
			nn->w_ih[i * nn->hidden_sz + j] += nn->delta_h[j] * in * lr;
	}

	// hidden bias
	for (size_t j = 0; j < nn->hidden_sz; j++)
		nn->b_h[j] += nn->delta_h[j] * lr;

	return loss / nn->output_sz;
}
