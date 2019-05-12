/*
 *   1drow.c
 *
 *   MPI code for performing parallel dense matrix-vector multiplication using 1D
 *   rowwise partitioning.
 *
 *   Written by cetinsamet -*- cetin.samet@metu.edu.tr
 *   April, 2019
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>


int main(int argc, char *argv[]) {

	int i, j;
    int rank, size;

    double minStart, maxEnd;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N 		= atoi(argv[1]);
    int pSize	= N / size;

    /* Initialize result vector */
    double* RESULT;
	RESULT = (double*) malloc(N * sizeof(double));

    /* Initialize matrix */
    double **matrix	= (double **) malloc(N * sizeof(double*));
	for(i = 0; i < N; i++)
		matrix[i] = (double *) malloc(N * sizeof(double));
    
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j)
			matrix[i][j] = i + j;
	}

	/* Initialize vector */
	double* vector;	
	vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < N; ++i)
		vector[i] = i;

	/* Initialize local matrix */
	double **p_matrix	= (double **) malloc(pSize * sizeof(double*));
	for(i = 0; i < pSize; i++)
		p_matrix[i] = (double *) malloc(N * sizeof(double));

	for (i = 0; i < pSize; ++i) {
		for (j = 0; j < N; ++j)
			p_matrix[i][j] = matrix[(rank * pSize) + i][j];
	}

	/* Initialize local vector */
	double* p_vector;
	p_vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < pSize; ++i)
		p_vector[(rank * pSize) + i] = vector[(rank * pSize) + i];

	double pStart = MPI_Wtime(); // <-- Start Time

    /* Broadcast all vector partitions to all other processes */
	for (i = 0; i < size; ++i) {
		for (j = 0; j < pSize; ++j)
			MPI_Bcast(&p_vector[(i * pSize) + j], 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
	}

    /* Calculate each independent dot product */
	for (i = 0; i < pSize; ++i)
		RESULT[(rank * pSize) + i] = cblas_ddot(N, p_matrix[i], 1, p_vector, 1);

    /* Broadcast all result partitions to all other processes */
	for (i = 0; i < size; ++i) {
		for (j = 0; j < pSize; ++j)
			MPI_Bcast(&RESULT[(i * pSize) + j], 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
	}

	double pEnd = MPI_Wtime(); // <-- Stop Measuring Time

	MPI_Reduce(&pStart,  &minStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pEnd, &maxEnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print results */
	if (rank == 0) {
		printf("RESULT of MATRIX-VECTOR MULTIPLICATION:\n");
		for (i = 0; i < N; ++i)
			printf("%f ", RESULT[i]);
		printf("\n");
		printf("Vector size: %d\tElapsed time: %f\n", N, maxEnd - minStart);
	}

	MPI_Finalize();
	return 0;
}
