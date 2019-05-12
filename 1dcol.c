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

    // ------------------------------------------------------- //
    // RESULT VECTOR    
    double* RESULT;
    RESULT 	= (double*) malloc(N * sizeof(double));

	// ------------------------------------------------------- //
	// LOCAL VECTOR    
	double* locResult;
	locResult = (double*) malloc(N * sizeof(double));

    // ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
    // MAIN MATRIX
    double **matrix	= (double **) malloc(N * sizeof(double*));
	for(i = 0; i < N; i++) {
		matrix[i] = (double *) malloc(N * sizeof(double));	
	}
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j){
			matrix[i][j] = i + j;
		}
	}

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// MAIN VECTOR
	double* vector;	
	vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < N; ++i) {
		vector[i] = i;
	}

	// ------------------------------------------------------- // 
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- // 
	// LOCAL MATRIX
	double **p_matrix	= (double **) malloc(N * sizeof(double*));
	for(i = 0; i < N; i++) {
		p_matrix[i] = (double *) malloc(pSize * sizeof(double));	
	}

	for (j = 0; j < pSize; ++j) {
		for (i = 0; i < N; ++i) {
			p_matrix[i][j] = matrix[i][(rank * pSize) + j];
		}
	}

	// ------------------------------------------------------- // 
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- // 
	// LOCAL VECTOR
	double* p_vector;
	p_vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < pSize; ++i) {
		p_vector[i] = vector[(rank * pSize) + i];
	}

	double pStart = MPI_Wtime(); // <-- Start Measuring Time

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// CALCULATE LOCAL DOT PRODUCT
	for (i = 0; i < pSize; ++i) {
		for (j = 0; j < N; ++j) {
			locResult[j] += p_matrix[j][i] * p_vector[i];
		}
	}
	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //

	for (i = 0; i < N; ++i) {
		MPI_Reduce(&locResult[i], &RESULT[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	
	}

	double pEnd = MPI_Wtime(); // <-- Stop Measuring Time

	MPI_Reduce(&pStart,  &minStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pEnd, &maxEnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// PRINT RESULTS
	if (rank == 0) {
		/*
		printf("RESULT of MATRIX-VECTOR MULTIPLICATION:\n");
		for (i = 0; i < N; ++i) {
			printf("%f ", RESULT[i]);
		}
		printf("\n");
		*/
		printf("Vector size: %d\tElapsed time: %f\n", N, maxEnd - minStart);
	}
	
	MPI_Finalize();
	return 0;
}
