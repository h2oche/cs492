#include "mmreader.hpp"
#include <time.h>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <pthread.h>

struct pthread_args {
    struct sparse_mtx* A;
    struct dense_mtx* B;
    struct dense_mtx* C;
    int startRow, endRow, load;
};

typedef std::vector<pthread_t> vth;
typedef std::vector<pthread_args> vthargs;

bool
SCsrMatrixfromFile(struct sparse_mtx *A, const char* filePath)
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
        {
            std::cout << "Reading file name error" << std::endl;
            return false;
        }
    }
    else
        return false;

    // Read data from a file on disk into buffers
    // Data is read natively as COO format with the reader
    MatrixMarketReader mm_reader;
    if( mm_reader.MMReadFormat(filePath) )
        return false;

    // JPA: Shouldn't that just be an assertion check? It seems to me that
    // the user have to call clsparseHeaderfromFile before calling this function,
    // otherwise the whole pCsrMatrix will be broken;
    A->nrow = mm_reader.GetNumRows( );
    A->ncol = mm_reader.GetNumCols( );
    A->nnze = mm_reader.GetNumNonZeroes( );

    A->row = (int32_t *)malloc((A->nrow + 1) * sizeof(int32_t));
    A->val = (float *)malloc(A->nnze * sizeof(float));
    A->col = (int32_t *)malloc(A->nnze * sizeof(int32_t));

    if(A->row == NULL || A->col == NULL || A->val == NULL)
    {
        if(A->row == NULL)
            free((void *)A->row);
        if(A->col == NULL)
            free((void *)A->col);
        if(A->val == NULL)
            free((void *)A->val);
        return false;
    }

    //  The following section of code converts the sparse format from COO to CSR
    Coordinate* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + A->nnze, CoordinateCompare );

    int32_t current_row = 1;

    A->row[ 0 ] = 0;

    for (int32_t i = 0; i < A->nnze; i++)
    {
        A->col[ i ] = coords[ i ].y;
        A->val[ i ] = coords[ i ].val;

        while( coords[ i ].x >= current_row )
            A->row[ current_row++ ] = i;
    }

    A->row[ current_row ] = A->nnze;

    while( current_row <= A->nrow )
        A->row[ current_row++ ] = A->nnze;

    return true;
}

void multiply_single(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C)
{
    // TODO: Implement matrix multiplication with single thread. C=A*B
    for(int i = 0 ; i < A->nrow; i ++) {
        int startColIdx = A->row[i];
        int endColIdx = i+1 < A->nrow ? A->row[i+1] : A->nrow;
        for(int j = 0 ; j < B->ncol; j ++) {
            for(int k = startColIdx ; k < endColIdx ; k++) {
                C->val[i * B->ncol + j] += A->val[k] * B->val[A->col[k] * B->ncol + j];
            }
        }
    }
}

static void* multiply_pthread_func(void* _args) {
    pthread_args* args = (pthread_args*)_args;

    int startRow = args->startRow;
    int endRow = args->endRow;

    for(int i = startRow ; i < endRow; i ++) {
        int startColIdx = args->A->row[i];
        int endColIdx = i+1 < args->A->nrow ? args->A->row[i+1] : args->A->nrow;
        for(int j = 0 ; j < args->B->ncol; j ++) {
            float sum = 0;
            for(int k = startColIdx ; k < endColIdx ; k++)
                sum += args->A->val[k] * args->B->val[args->A->col[k] * args->B->ncol + j];
            args->C->val[i * args->B->ncol + j] = sum;
        }
    }
}

void multiply_pthread(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C, int nThread)
{
    // TODO: Implement matrix multiplication with pthread. C=A*B

    /* initialize thread argument */
    vth threads = vth(nThread);
    vthargs threadArgs = vthargs(nThread);
    int status;

    for(int i = 0 ; i < nThread ; i ++) {
        threadArgs[i].A = A;
        threadArgs[i].B = B;
        threadArgs[i].C = C;
    }

    /* load balancing */
    int threshold = A->nnze / nThread;
    int load = 0;
    int threadIdx = 0;
    bool isStartSet = false;
    for(int i = 0 ; i < A->nrow ; i ++) {
        if(!isStartSet) {
            threadArgs[threadIdx].startRow = i;
            isStartSet = true;
        }

        int loadRow = (i+1 < A->nrow ? A->row[i+1] : A->nrow) - A->row[i];
        load += loadRow;
        if(load >= threshold) {
            threadArgs[threadIdx].endRow = i+1;
            threadArgs[threadIdx].load = load;
            threadIdx++;
            load = 0;
            isStartSet = false;
        }
    }
    threadArgs.back().load = A->nnze;
    for(int i = 0 ; i < nThread - 1 ; i ++)
        threadArgs.back().load -= threadArgs[i].load;
    threadArgs.back().endRow = A->nrow;

    std::cout << "total load : " << A->nnze << " with threshold : " << threshold << std::endl;
    for(int i = 0 ; i < nThread ; i ++) {
        std::cout << "start row : " << threadArgs[i].startRow << ", end row : " << threadArgs[i].endRow << " with load " << threadArgs[i].load << std::endl;
    }

    /* computation */
    for(int i = 0 ;i < nThread ; i ++)
        pthread_create(&threads[i], NULL, multiply_pthread_func, &threadArgs[i]);
    for(int i = 0 ; i < nThread ; i++)
        pthread_join(threads[i], (void**)&status);
}

uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

int main(int argc, char **argv)
{
    int nThread = atoi(argv[3]);

    struct sparse_mtx A;
    if(!SCsrMatrixfromFile(&A, argv[1]))
    {
        std::cout << "read failed." << std::endl;
        return 0;
    }

    struct dense_mtx B;
    B.nrow = A.ncol;
    B.ncol = atoi(argv[2]);
    if(B.ncol < 0)
    {
        free(A.row);
        free(A.col);
        free(A.val);
        std::cerr << "Invalid argument for the number of columns of B." << std::endl;
    }
    B.val = (float *)malloc(sizeof(float) * B.nrow * B.ncol);

    srand((unsigned int)time(NULL));
    for(int i = 0; i < B.nrow; i++)
    {
        for(int j = 0; j < B.ncol; j++)
        {
            B.val[B.ncol * i + j] = ((float)rand()/(float)(RAND_MAX)) * ((rand() % 2) ? 1.0f : -1.0f);
        }
    }

    struct dense_mtx C1, C2;
    C1.val = NULL;
    C2.val = NULL;

    C1.nrow = A.nrow, C1.ncol = B.ncol;
    C2.nrow = A.nrow, C2.ncol = B.ncol;
    C1.val = (float *)malloc(sizeof(float) * A.nrow * B.ncol);
    C2.val = (float *)malloc(sizeof(float) * A.nrow * B.ncol);

    std::cout << "Single Thread Computation Start" << std::endl;
    uint64_t start = GetTimeStamp();
    multiply_single(&A, &B, &C1);
    uint64_t end = GetTimeStamp();
    std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;
    std::cout << "Multi Thread Computation Start" << std::endl;
    start = GetTimeStamp();
    multiply_pthread(&A, &B, &C2, nThread);
    end = GetTimeStamp();
    std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

    // TODO: Testing Code by comparing C1 and C2
    double error = 0;
    error = 0;
    for(int i = 0 ; i < C1.nrow * C1.ncol ; i ++)
        error += abs(C1.val[i] - C2.val[i]);
    std::cout << "error : " << error << std::endl;

    free(A.row);
    free(A.col);
    free(A.val);
    free(B.val);
    if(C1.val != NULL)
        free(C1.val);
    if(C2.val != NULL)
        free(C2.val);

    return 0;
}
