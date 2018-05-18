#include <stdio.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <FLANN/util/params.h>
#include <FLANN/io/hdf5.h>
#include <FLANN/mpi/client.h>



#define IF_RANK0 if (world.rank()==0)

timeval start_time_;
void start_timer(const std::string& message = "")
{
	if (!message.empty()) {
		printf("%s", message.c_str());
		fflush(stdout);
	}
    gettimeofday(&start_time_,NULL);
}

double stop_timer()
{
    timeval end_time;
    gettimeofday(&end_time,NULL);

	return double(end_time.tv_sec-start_time_.tv_sec)+ double(end_time.tv_usec-start_time_.tv_usec)/1000000;
}

float compute_precision(const flann::Matrix<int>& match, const flann::Matrix<int>& indices)
{
	int count = 0;

	assert(match.rows == indices.rows);
	size_t nn = std::min(match.cols, indices.cols);

	for(size_t i=0; i<match.rows; ++i) {
		for (size_t j=0;j<nn;++j) {
			for (size_t k=0;k<nn;++k) {
				if (match[i][j]==indices[i][k]) {
					count ++;
				}
			}
		}
	}

	return float(count)/(nn*match.rows);
}


int main(int argc, char* argv[])
{
	try {

		flann::Matrix<float> query;
		flann::Matrix<int> match;

		flann::load_from_file(query, "sift100K.h5","query");
		flann::load_from_file(match, "sift100K.h5","match");
		//	flann::load_from_file(gt_dists, "sift100K.h5","dists");

		flann::mpi::Client index("localhost","9999");

		int nn = 1;
		flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
		flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

		start_timer("Performing search...\n");
		index.knnSearch(query, indices, dists, nn, flann::SearchParams(64));
		printf("Search done (%g seconds)\n", stop_timer());

		printf("Checking results\n");
		float precision = compute_precision(match, indices);
		printf("Precision is: %g\n", precision);

	}
	catch (std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}

