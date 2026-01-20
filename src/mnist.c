#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

MNIST_Dataset *mnist_load(const char *img_filename, const char *labels_filename) {
	FILE *img_f = fopen(img_filename, "rb");
	FILE *lbl_f = fopen(labels_filename, "rb");

	uint32_t magic, count, rows, cols;
	uint32_t lbl_magic, lbl_count;

	fread(&magic, 4, 1, img_f);
	fread(&count, 4, 1, img_f);
	fread(&rows, 4, 1, img_f);
	fread(&cols, 4, 1, img_f);

	fread(&lbl_magic, 4, 1, lbl_f);
	fread(&lbl_count, 4, 1, lbl_f);

	magic = big2little(magic);
	count = big2little(count);
	rows = big2little(rows);
	cols = big2little(cols);

	lbl_magic = big2little(lbl_magic);
	lbl_count = big2little(lbl_count);

	printf("Read %d images %dx%d\nMagic: %d | Labels: %d\n", count, rows, cols, magic, lbl_count);
	
	MNIST_Dataset *dataset = malloc(sizeof(MNIST_Dataset));

	dataset->img_count = count;
	dataset->rows = rows;
	dataset->cols = cols;
	dataset->images = malloc(count * rows * cols);
	dataset->labels = malloc(lbl_count);

	// Reading the images and labels
	fread(dataset->images, 1, count * rows * cols, img_f);
	fread(dataset->labels, 1, lbl_count, lbl_f);

	fclose(img_f);
	fclose(lbl_f);

	return dataset;
}

void mnist_free(MNIST_Dataset *dataset) {
	if (dataset != NULL) {
		if (dataset->images != NULL) free(dataset->images);
		if (dataset->labels != NULL) free(dataset->labels);
		free(dataset);
	}
}
