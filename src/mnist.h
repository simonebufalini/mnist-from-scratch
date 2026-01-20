#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

static inline uint32_t big2little(uint32_t num) {
	return	((num>>24)	&	0x000000ff) | 	// move byte 3 to byte 0
            ((num>>8)	&	0x0000ff00) | 	// move byte 1 to byte 2
            ((num<<8)	&	0x00ff0000) | 	// move byte 2 to byte 1
            ((num<<24)	&	0xff000000); 	// byte 0 to byte 3
}

typedef struct {
	uint32_t img_count;		// Number of images
	uint32_t rows;			// Rows per image
	uint32_t cols;			// Columns per image
	uint8_t *images;		// Pointer to images
	uint8_t *labels;		// Pointer to labels
} MNIST_Dataset;

MNIST_Dataset *mnist_load(const char *img_filename, const char *labels_filename);
void mnist_free(MNIST_Dataset *dataset);

#endif
