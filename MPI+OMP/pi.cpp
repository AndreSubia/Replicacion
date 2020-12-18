#include <mpi.h>
#include <iostream>
#include <omp.h>

using namespace std;

constexpr int kMaster = 0;
constexpr long kIteration = 1024;
constexpr long kScale = 45;
constexpr long kTotalNumStep = kIteration * kScale;


void CalculatePiParallel(float* results, int rank_num, int num_procs);

int main(int argc, char* argv[]) {
    int i, id, num_procs;
    float total_pi;
    MPI_Status stat;

    // Start MPI.
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        cout << "Failed to initialize MPI\n";
        exit(-1);
    }
    
    // Create the communicator, and retrieve the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Determine the rank of the process.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int num_step_per_rank = kTotalNumStep / num_procs;
    float* results_per_rank = new float[num_step_per_rank];
    for (size_t i = 0; i < num_step_per_rank; i++) results_per_rank[i] = 0.0;

    // Calculate the Pi number partially in parallel.
    CalculatePiParallel(results_per_rank, id, num_procs);

    float sum = 0.0;
    for (size_t i = 0; i < num_step_per_rank; i++) sum += results_per_rank[i];

    delete[] results_per_rank;

    MPI_Reduce(&sum, &total_pi, 1, MPI_FLOAT, MPI_SUM, kMaster, MPI_COMM_WORLD);

    if (id == kMaster) cout << "---> pi= " << total_pi << "\n";

    MPI_Finalize();

    return 0;
}


void CalculatePiParallel(float* results, int rank_num, int num_procs) {
    char machine_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    int is_cpu=true;
    int num_step = kTotalNumStep / num_procs;
    float* x_pos_per_rank = new float[num_step];
    float dx, dx_2;

    // Get the machine name.
    MPI_Get_processor_name(machine_name, &name_len);

    dx = 1.0f / (float)kTotalNumStep;
    dx_2 = dx / 2.0f;

    for (size_t i = 0; i < num_step; i++)
        x_pos_per_rank[i] = ((float)rank_num / (float)num_procs) + i * dx + dx_2;
    
    #pragma omp target map(from:is_cpu) map(to:x_pos_per_rank[0:num_step]) map(from:results[0:num_step])
    {  
        #pragma omp teams distribute parallel for simd
        // Use loop to calculate a partial of the number Pi in parallel.
        for (int k=0; k< num_step; k++) {
            if (k==0) is_cpu=omp_is_initial_device();
            float x = x_pos_per_rank[k];
            results[k] = (4.0f * dx) / (1.0f + x * x);
        }
    }
    cout << "Rank " << rank_num << " of " << num_procs
         << " runs on: " << machine_name
         << ", uses device: " << (is_cpu?"CPU":"GPU")
         << "\n";

    // Cleanup.
    delete[] x_pos_per_rank;
}