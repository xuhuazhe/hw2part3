//
// Created by An Ju on 2/28/18.
//

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <assert.h>

#include "common.h"

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

int get_ind(particle_t &p, int width) {
    return floor(p.x / cutoff) + width * floor(p.y / cutoff);
}

//
//  benchmarking program
//
int main( int argc, char **argv ) {
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 ) {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    //particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    particle_t* particles = new particle_t[n];
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    // Set up bins
    double size = sqrt( density * n );
    int width = (int)ceil(size / cutoff);
    int numbins = width * width;
    std::vector< std::vector<particle_t> > bins;
    bins.resize(numbins);

    // Assign particles to bins
    for (int i = 0; i != n; i++) {
        bins[get_ind(particles[i], width)].push_back(particles[i]);
    }

    // Calculate the range of bins in charge
    // Each process controls the same number of rows
    int num_rows = width / n_proc;
    int row_start = num_rows * rank;
    int row_end = num_rows * (rank + 1);
    if (rank == n_proc - 1) {
        row_end = width;
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );

        //
        //  compute all forces
        //
        for (int i = row_start; i < row_end; i++) {
            for (int j = 0; j < width; j++) {
                std::vector<particle_t>& tmp_bin = bins[i*width+j];
                for (int p = 0; p < tmp_bin.size(); p++) {
                    tmp_bin[p].ax = tmp_bin[p].ay = 0;

                    int ind = get_ind(tmp_bin[p], width);
                    int i_min = ind < width ? 0 : -1;
                    int i_max = ind >= numbins - width ? 0 : 1;
                    int j_min = ind % width == 0 ? 0 : -1;
                    int j_max = (ind + 1) % width == 0 ? 0 : 1;

                    for (int ii = i_min; ii <= i_max; ii++ ) {
                        for (int jj = j_min; jj <= j_max; jj++) {
                            int bin_i = ind + jj + ii * width;
                            for (int k = 0; k != bins[bin_i].size(); k++) {
                                apply_force(tmp_bin[p], bins[bin_i][k], &dmin, &davg, &navg);
                            }
                        }
                    }
                }
            }
        }

        //
        // move particles
        //
        std::vector<particle_t> message_1, message_2;
        std::vector<particle_t> local_cache;
        if (rank != 0) {
            for (int i = 0; i < width; i++) {
                bins[(row_start-1)*width+i].clear();
            }
        }
        if (rank != n_proc-1) {
            for (int i = 0; i < width; i++) {
                bins[row_end*width+i].clear();
            }
        }
        for (int i = row_start; i < row_end; i++) {
            for (int j = 0; j < width; j++) {
                std::vector<particle_t>& tmp_bin = bins[i*width+j];
                for (int p = 0; p < tmp_bin.size(); p++) {
                    move(tmp_bin[p]);
                    int bin_row = (int)floor(tmp_bin[p].y / cutoff);

                    if (bin_row <= row_start) {
                        message_1.push_back(tmp_bin[p]);
                    }
                    if (bin_row >= (row_end-1)) {
                        message_2.push_back(tmp_bin[p]);
                    }
                    local_cache.push_back(tmp_bin[p]);
                }
                tmp_bin.clear();
            }
        }
        for (int p = 0; p < local_cache.size(); p++) {
            bins[get_ind(local_cache[p], width)].push_back(local_cache[p]);
        }

        MPI_Request req_1, req_2;
        std::vector<particle_t> cmessage_1, cmessage_2;
        // Send message upward
        if (rank != 0) {
            cmessage_1.reserve(message_1.size());
            cmessage_1.insert(cmessage_1.end(), message_1.begin(), message_1.end());
            message_1.clear();
            MPI_Isend(cmessage_1.data(), int(cmessage_1.size()), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &req_1);
        }
        if (rank != n_proc - 1) {
            cmessage_2.reserve(message_2.size());
            cmessage_2.insert(cmessage_2.end(), message_2.begin(), message_2.end());
            message_2.clear();
            MPI_Isend(cmessage_2.data(), int(cmessage_2.size()), PARTICLE, rank+1, 0, MPI_COMM_WORLD, &req_2);
        }

        MPI_Status status_1, status_2;
        int size_1, size_2;
        std::vector<particle_t> msg_1, msg_2;
        if (rank != 0) {
            MPI_Probe(rank-1, 0, MPI_COMM_WORLD, &status_1);
            MPI_Get_count(&status_1, PARTICLE, &size_1);
            msg_1.resize(size_1);
            MPI_Recv(msg_1.data(), size_1, PARTICLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int p = 0; p < msg_1.size(); p++) {
                bins[get_ind(msg_1[p], width)].push_back(msg_1[p]);
            }
        }
        if(rank != n_proc-1) {
            MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &status_2);
            MPI_Get_count(&status_2, PARTICLE, &size_2);
            msg_2.resize(size_2);
            MPI_Recv(msg_2.data(), size_2, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int p = 0; p < msg_2.size(); p++) {
                bins[get_ind(msg_2[p], width)].push_back(msg_2[p]);
            }
        }

        MPI_Barrier( MPI_COMM_WORLD );

        if( find_option( argc, argv, "-no" ) == -1 )
        {

            MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);


            if (rank == 0){
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg +=  rdavg/rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }

    }
    simulation_time = read_timer( ) - simulation_time;

    if (rank == 0) {
        printf( "n = %d, simulation time = %g seconds", n, simulation_time);

        if( find_option( argc, argv, "-no" ) == -1 )
        {
            if (nabsavg) absavg /= nabsavg;
            //
            //  -The minimum distance absmin between 2 particles during the run of the simulation
            //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
            //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
            //
            printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
            if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");

        //
        // Printing summary data
        //
        if( fsum)
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }

    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
//    free( partition_offsets );
//    free( partition_sizes );
//    free( local );
    free(particles);
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}
