#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <Windows.h>
#include <math.h>
#include <WinBase.h>


// Use to convert Bytes to KB
#define DIV 1024
int max_threads = 0;

float** initMatrix(unsigned int dimension) {
	//init matrix
	float** matrix = malloc(dimension * sizeof(float*));
	int i;
	#pragma omp parallel for
	for (i = 0; i < dimension; i++) {
		matrix[i] = malloc(dimension * sizeof(float));
		for (unsigned int j = 0; j < dimension; j++) {
			matrix[i][j] = (float)rand() / (float)(RAND_MAX);
		}
	}
	return matrix;
}

float* initVektor(unsigned int dimension) {
	//init vector
	float* vector = malloc(dimension * sizeof(float*));
	for (unsigned int i = 0; i < dimension; i++) {
		vector[i] = (float)rand() / (float)(RAND_MAX);
	}
	return vector;
}


float* multiplyMatrix(float** matrix, float* vector, unsigned long dimension, int numOfThreads) {
	double start;
	double end;
	int i;
	omp_set_num_threads(numOfThreads);
	float* result = malloc(dimension * sizeof(float*));
	for (unsigned int k = 0; k < dimension; k++) {
		result[k] = 0.0f;
	}

	start = omp_get_wtime();

	#pragma omp parallel for		
	for (i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			result[i] += matrix[i][j] * vector[j];
		}
	}
	end = omp_get_wtime(); 
	printf("NumOfThreads: %d // Dimension: %lu // Work took %f seconds \n", numOfThreads, dimension, end - start);
	return result;
}

float* multiplyMatrixSchedule(float** matrix, float* vector, unsigned long dimension, int numOfThreads, int schedule, int c) {
	double start;
	double end;
	int i;
	omp_set_num_threads(numOfThreads);
	float* result = malloc(dimension * sizeof(float*));
	for (unsigned int k = 0; k < dimension; k++) {
		result[k] = 0.0f;
	}

	start = omp_get_wtime();
	switch (schedule)	{
		case 1:
		printf("Dynamic; %d //", c);
		#pragma omp parallel for schedule(dynamic, c)
		for (i = 0; i < dimension; i++) {
			for (unsigned int j = 0; j < dimension; j++) {
				float temp = matrix[i][j] * vector[j];
				result[i] += temp;
				unsigned int k;
				for (k = 0; k < (int)temp * 10000000; k++) {}
			}
		}
		break;
		case 2:
		printf("Static; %d //", c);
		#pragma omp parallel for schedule(static, c)
		for (i = 0; i < dimension; i++) {
			for (unsigned int j = 0; j < dimension; j++) {
				float temp = matrix[i][j] * vector[j];
				result[i] += temp;
				unsigned int k;
				for (k = 0; k < (int)temp * 10000000; k++) {}
			}
		}
		break;
		case 3:
		printf("Guided; %d //", c);
		#pragma omp parallel for schedule(guided, c)
		for (i = 0; i < dimension; i++) {
			for (unsigned int j = 0; j < dimension; j++) {
				float temp = matrix[i][j] * vector[j];
				result[i] += temp;
				unsigned int k;
				for (k = 0; k < (int)temp * 10000000; k++) {}
			}
		}
		break;
		case 4:
		printf("Runtime; %d //", c);
		#pragma omp parallel for schedule(runtime)
		for (i = 0; i < dimension; i++) {
			for (unsigned int j = 0; j < dimension; j++) {
				float temp = matrix[i][j] * vector[j];
				result[i] += temp;
				unsigned int k;
				for (k = 0; k < (int)temp * 10000000; k++) {}
			}
		}
		break;
		}
	end = omp_get_wtime(); 
	printf("NumOfThreads: %d // Dimension: %lu // Work took %f seconds \n", numOfThreads, dimension, end - start);
	return result;
}

float* multiplyMatrixSchedule2(float** matrix, float* vector, unsigned long dimension, int numOfThreads, LPCWSTR schedule) {
	double start;
	double end;
	int i;
	omp_set_num_threads(numOfThreads);
	SetEnvironmentVariable(TEXT("OMP_SCHEDULE"), schedule);
	float* result = malloc(dimension * sizeof(float*));
	for (unsigned int k = 0; k < dimension; k++) {
		result[k] = 0.0f;
	}

	start = omp_get_wtime();
	#pragma omp parallel for schedule(runtime)
	for (i = 0; i < dimension; i++) {
		for (unsigned int j = 0; j < dimension; j++) {
			float temp = matrix[i][j] * vector[j];
			result[i] += temp;
			unsigned int k;
			for (k = 0; k < (int)temp * 10000000; k++) {}
		}
	}

	end = omp_get_wtime();
	printf("Schedule: %ls // NumOfThreads: %d // Dimension: %lu // Work took %f seconds \n", schedule, numOfThreads, dimension, end - start);
	return result;
}


unsigned int getMaxRamDimension() {
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof(statex);
	GlobalMemoryStatusEx(&statex);

	unsigned long long totalRAM = statex.ullTotalPhys;
	unsigned long long freeRAM = statex.ullAvailPhys; //bytes

	//SAFETY FEATURE
	unsigned long long maxRAMSize = freeRAM* 0.9;
	//Count of floats fits to memory
	unsigned long long maxFloatNumber = maxRAMSize / sizeof(float);
	//Quadratische Ergenzung immer gleich n^2 + 2n
	unsigned long long maxVectorLength = (unsigned long long)(sqrt(maxFloatNumber + 1) - 1);

	printf("--------------------------------------------------\n");
	printf("There are %llu total GB of physical Memory.\n", totalRAM / DIV / DIV/ DIV);
	printf("There are %llu free GB of physical Memory.\n", freeRAM / DIV / DIV / DIV);
	printf("There are %llu free GB of max physical Memory.\n", maxRAMSize / DIV / DIV / DIV);
	printf("Max usable Floats: %llu\n", maxFloatNumber);
	printf("Max Vector Size: %llu\n", maxVectorLength);
	printf("Max used Floats: %llu\n",(unsigned long long)pow(maxVectorLength, 2) + 2*maxVectorLength);
	printf("--------------------------------------------------\n");
	
	return maxVectorLength;
}

int main(void) {
	double start;
	double end;
	max_threads = omp_get_max_threads();
	//get max possible array size
	unsigned int maxRamDimension = getMaxRamDimension();


	start = omp_get_wtime();
	//init array and vector
	float** matrix = initMatrix(maxRamDimension);
	float* vector = initVektor(maxRamDimension);
	float* result = malloc(maxRamDimension * sizeof(float*));
	end = omp_get_wtime();
	printf("--------------------------------------------------\n");
	printf("Initialization took %f seconds\n", end - start);
	printf("Start Calculation:\n");
	printf("--------------------------------------------------\n");
	result = multiplyMatrix(matrix, vector, 1000, 1);
	result = multiplyMatrix(matrix, vector, 1000, max_threads);
	result = multiplyMatrix(matrix, vector, 10000, 1);
	result = multiplyMatrix(matrix, vector, 10000, max_threads);
	result = multiplyMatrix(matrix, vector, 30000, 1);
	result = multiplyMatrix(matrix, vector, 30000, max_threads);
	result = multiplyMatrix(matrix, vector, 50000, 1);
	result = multiplyMatrix(matrix, vector, 50000, max_threads);
	result = multiplyMatrix(matrix, vector, maxRamDimension, 1);
	result = multiplyMatrix(matrix, vector, maxRamDimension, max_threads);
	printf("--------------------------------------------------\n");
	printf("Schedule Change:\n");
	printf("--------------------------------------------------\n");
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 1, 1);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("dynamic,1"));
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 2, 1);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("static,1"));
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 3, 1);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("guided,1"));
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 1, 10);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("dynamic,10"));
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 2, 10);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("static,10"));
	result = multiplyMatrixSchedule(matrix, vector, maxRamDimension, max_threads, 3, 10);
	result = multiplyMatrixSchedule2(matrix, vector, maxRamDimension, max_threads, TEXT("guided,10"));

	return 0;
}