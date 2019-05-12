/*
 *   1dcol.c
 *
 *   MPI code for performing parallel dense matrix-vector multiplication using 1D
 *   columnwise partitioning.
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
    RESULT 	= (double*) malloc(N * sizeof(double));

	/* Initialize local result vector */
	double* locResult;
	locResult = (double*) malloc(N * sizeof(double));

    /* Initialize matrix */
    double **matrix	= (double **) malloc(N * sizeof(double*));
	for(i = 0; i < N; i++) {
		matrix[i] = (double *) malloc(N * sizeof(double));	
	}
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j)
			matrix[i][j] = i + j;
	}

	/* Initialize vector */
	double* vector;	
	vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < N; ++i)
		vector[i] = i;

	/* Initialize matrix */
	double **p_matrix	= (double **) malloc(N * sizeof(double*));
	for(i = 0; i < N; i++)
		p_matrix[i] = (double *) malloc(pSize * sizeof(double));

	for (j = 0; j < pSize; ++j) {
		for (i = 0; i < N; ++i)
			p_matrix[i][j] = matrix[i][(rank * pSize) + j];
	}

	/* Initialize local vector */
	double* p_vector;
	p_vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < pSize; ++i) {
		p_vector[i] = vector[(rank * pSize) + i];
	}

	double pStart = MPI_Wtime(); // <-- Start Time

	/* Calculate local dot product */
	for (i = 0; i < pSize; ++i) {
		for (j = 0; j < N; ++j) {
			locResult[j] += p_matrix[j][i] * p_vector[i];

	for (i = 0; i < N; ++i)
		MPI_Reduce(&locResult[i], &RESULT[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	double pEnd = MPI_Wtime(); // <-- Stop Time

	MPI_Reduce(&pStart,  &minStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pEnd, &maxEnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Print result */
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
