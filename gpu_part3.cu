#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <vector>
#include "common.h"
using std::vector;

#define NUM_THREADS 256

extern double size;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

__global__ void assign_particle_to_bin(int n, particle_t* d_particles,
    int width, int maxnum_per_bin, int* bin_count, particle_t** bins) {
  CUDA_KERNEL_LOOP (i, n) {
    particle_t* p = &d_particles[i];
    int bin_ind = floor(p->x/cutoff) + width*floor(p->y/cutoff);

    // Put particle in bin_ind
    // Compute the position of particle inside the bin with AtomicAdd
    // to avoid race.
    int list_ind = atomicAdd(&bin_count[bin_ind], 1);  // index within bin

    // Check if this bin is full
    if (list_ind >= maxnum_per_bin) {  // bin full (very unlikely but possible)
      atomicAdd(&bin_count[bin_ind], -1);  // reverse the effect of previous atomicAdd

      // "outsider collector" is at the end and can contain enough particles
      bin_ind = width*width;
      // index in "outsider collector"
      list_ind = atomicAdd(&bin_count[bin_ind], 1);
    }
    bins[bin_ind*maxnum_per_bin + list_ind] = p;
  }
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(int n, particle_t* d_particles,
    int width, int maxnum_per_bin, int* bin_count, particle_t** bins) {
  CUDA_KERNEL_LOOP (i, n) {
    particle_t* p = &d_particles[i];
    p->ax = p->ay = 0;

    int ind = floor(p->x/cutoff) + width*floor(p->y/cutoff);

    int i_min = ind < width ? 0 : -1;
    int i_max = ind >= width*width - width ? 0 : 1;
    int j_min = ind % width == 0 ? 0 : -1;
    int j_max = (ind + 1) % width == 0 ? 0 : 1;

    // apply nearby forces
    // Note: no need to skip itself in the bin
    for (int i = i_min; i <= i_max; i++) {
      for (int j = j_min; j <= j_max; j++) {
        int bin_i = ind + j + width*i;
        for (int k = 0; k < bin_count[bin_i]; k++ ) {
          apply_force_gpu(*p, *bins[bin_i*maxnum_per_bin+k]);
        }
      }
    }

    // collect force from particles in "outsider collector"
    int nbin = width * width;
    for (int k = 0; k < bin_count[nbin]; k++ )
      apply_force_gpu(*p, *bins[nbin*maxnum_per_bin+k]);
  }
}

__global__ void move_gpu (particle_t * d_particles, int n, double size) {
  CUDA_KERNEL_LOOP(i, n) {
    particle_t * p = &d_particles[i];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
  }
}

//  benchmarking program
//
int main( int argc, char **argv )
{
   // This takes a few seconds to initialize the runtime
   cudaThreadSynchronize();

   if( find_option( argc, argv, "-h" ) >=0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    // create spatial bins (of size cutoff by cutoff)
    // the maximum possible numbers of particles inside a bin
    // Note: if there happen to be more particles in that bin, put it in "outsider collector"
    const int maxnum_per_bin = 5;
    // create an extra bin as "outsider collector"
    double size = sqrt( density * n );
    int width = ceil(size/cutoff);
    int numbins = width*width;

    // Bins for particles
    // bins will be a (width, width, maxnum_per_bin) array
    int * bin_count;
    particle_t ** bins;
    // Memory allocation
    size_t bin_count_memsize = (numbins + 1) * sizeof(int);
    if (cudaSuccess != cudaMalloc((void **) &bin_count, bin_count_memsize)) {
      printf("ERROR: failed to allocate GPU array bin_count of size %lu\n", bin_count_memsize);
      return -1;
    }
    size_t bins_memsize = (numbins*maxnum_per_bin + 1) * sizeof(particle_t*);
    if (cudaSuccess != cudaMalloc((void **) &bins, bins_memsize)) {
      printf("ERROR: failed to allocate GPU array bins of size %lu\n", bins_memsize);
      return -1;
    }

    unsigned int total_memsize = bin_count_memsize + bins_memsize;
    printf("GPU allocation Bytes: %lu (= %f MB)\n", total_memsize, total_memsize / 1048576.0);

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
      // clear bins at each time step
      // Mark all bins as "no particle"
      // Also clear the "outsider collector" (extra bin)
      cudaMemset(bin_count, 0, (numbins + 1) * sizeof(int));


      // place particles in bins
      int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
      assign_particle_to_bin<<<blks, NUM_THREADS>>>(n, d_particles, width,
          maxnum_per_bin, bin_count, bins);

      //
      //  compute forces
      //
      compute_forces_gpu<<<blks, NUM_THREADS>>>(n, d_particles, width,
          maxnum_per_bin, bin_count, bins);

      move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

      if( fsave && (step%SAVEFREQ) == 0 ) {
        // Copy the particles back to the CPU
        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
        save( fsave, n, particles);
      }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    cudaFree(bin_count);
    cudaFree(bins);
    if( fsave )
        fclose( fsave );
    // if (cpu_check) {
    //   delete[] bins;
    //   free(check_particles);
    // }
    return 0;
}
