#include "mnist.h"
#include "nn.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define EPOCHS 10
#define LR 0.1f

#define INPUT 784
#define HIDDEN 128
#define OUTPUT 10

const char images_filename[] = "../dataset/training/train-images-idx3-ubyte";
const char labels_filename[] = "../dataset/training/train-labels-idx1-ubyte";

const char test_images[] = "../dataset/test/t10k-images-idx3-ubyte";
const char test_labels[] = "../dataset/test/t10k-labels-idx1-ubyte";

// fischer-yates algorithm to shuffle the array
void shuffle(int *array, size_t n) {
	for (size_t i = n-1; i > 0; i--) {
		size_t j = rand() % (i+1);
		int t = array[i];
		array[i] = array[j];
		array[j] = t;
	}
}

// Argmax - find the actual prediction
int argmax(NN *nn) {
	int max_idx = 0;
	float max_val = nn->o_layer[0];
	
	for (int i = 0; i < nn->output_sz; i++) {
		if (nn->o_layer[i] > max_val) {
			max_val = nn->o_layer[i];
			max_idx = i;
		}
	}

	return max_idx;
}

int main(void) {
    srand(time(NULL));

    printf("Loading MNIST training dataset...\n");
	MNIST_Dataset *data = mnist_load(images_filename, labels_filename);

	printf("Loading MNIST testing dataset...\n");
	MNIST_Dataset *test = mnist_load(test_images, test_labels);

	if (data == NULL || test == NULL) {
        fprintf(stderr, "Error loading datasets\n");
        exit(EXIT_FAILURE);
		// use a more specific error ?
    }

	printf("Training images: %d\n", data->img_count);

	NN *net = nn_init(INPUT, HIDDEN, OUTPUT);
	if (net == NULL) {
		fprintf(stderr, "Error initializing network\n");
		exit(EXIT_FAILURE);
	}

	// array index for the shiffle
	int *idxs = malloc(data->img_count * sizeof(*idxs));
	for (int i = 0; i < data->img_count; i++) idxs[i] = i;

	float *input_img = malloc(INPUT * sizeof(*input_img));

	// TRAINING LOOP
	for (int epoch = 0; epoch < EPOCHS; epoch++) {

		printf("\n--- EPOCH %d/%d ---\n", epoch+1, EPOCHS);
		shuffle(idxs, data->img_count);
		float epoch_loss = 0.0f;

		clock_t start = clock();

		// looping on the training set
		for (int i = 0; i < data->img_count; i++) {
			int idx = idxs[i];	// select image
			
			// normalization from uint8_t to float (0.0 - 1.0)
			for (int k = 0; k < INPUT; k++)
				input_img[k] = data->images[idx * INPUT + k] / 255.0f;

			int target = data->labels[idx];
			float loss = nn_train(net, input_img, target, LR);
			epoch_loss += loss;

			if ((i+1) % 15000 == 0) {
				printf("\rEpoch %d: %d/%d images processed...", epoch+1, i+1, data->img_count);
				fflush(stdout);
			}
		}

		double t = (double)(clock() - start) / CLOCKS_PER_SEC;
		float avg_loss = epoch_loss / data->img_count;
		
		// evaluating accuracy
		int correct = 0;
		for (int i = 0; i < test->img_count; i++) {
			
			for (int k = 0; k < INPUT; k++)
				input_img[k] = test->images[i * INPUT + k] / 255.0f;

			nn_forward(net, input_img);

			if (argmax(net) == test->labels[i])
				correct ++;
		}

		float accuracy = (float)correct / test->img_count * 100.0f;
		printf("Time: %.2f | Loss: %.5f | Accuracy: %.2f%%\n", t, avg_loss, accuracy);
	}	// end of the epoch

	free(idxs);
	free(input_img);
	nn_free(net);
	mnist_free(data);
	mnist_free(test);

    return 0;
}
