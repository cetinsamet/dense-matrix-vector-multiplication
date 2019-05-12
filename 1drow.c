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
	RESULT = (double*) malloc(N * sizeof(double));

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
	double **p_matrix	= (double **) malloc(pSize * sizeof(double*));
	for(i = 0; i < pSize; i++) {
		p_matrix[i] = (double *) malloc(N * sizeof(double));	
	}

	for (i = 0; i < pSize; ++i) {
		for (j = 0; j < N; ++j) {
			p_matrix[i][j] = matrix[(rank * pSize) + i][j];
		}
	}

	// ------------------------------------------------------- // 
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- // 
	// LOCAL VECTOR
	double* p_vector;
	p_vector = (double*) malloc(N * sizeof(double));
	for (i = 0; i < pSize; ++i) {
		p_vector[(rank * pSize) + i] = vector[(rank * pSize) + i];
	}

	double pStart = MPI_Wtime(); // <-- Start Measuring Time

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// BROADCAST ALL VECTOR PARTITIONS TO ALL PROCESSES
	for (i = 0; i < size; ++i) {
		for (j = 0; j < pSize; ++j) {
			MPI_Bcast(&p_vector[(i * pSize) + j], 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
		}
	}

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// CALCULATE EACH INDEPENDENT DOT PRODUCT (MATRIX ROW TIMES VECTOR)
	for (i = 0; i < pSize; ++i) {
		RESULT[(rank * pSize) + i] = cblas_ddot(N, p_matrix[i], 1, p_vector, 1);
	}

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	// BROADCAST ALL RESULT PARTITIONS TO ALL PROCESSES
	for (i = 0; i < size; ++i) {
		for (j = 0; j < pSize; ++j) {
			MPI_Bcast(&RESULT[(i * pSize) + j], 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
		}
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

	// ------------------------------------------------------- //
	// * ----- * ----- * ----- * ----- * ----- * ----- * ----- * 
	// ------------------------------------------------------- //
	
	MPI_Finalize();
	return 0;
}
