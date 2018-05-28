#include <boost/mpi.hpp>
#include <FLANN/mpi/server.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);

	try {
		if (argc != 4) {
			std::cout << "Usage: " << argv[0] << " <file> <dataset> <port>\n";
			return 1;
		}
		flann::mpi::Server<flann::L2<float> > server(argv[1], argv[2], std::atoi(argv[3]),
				flann::KDTreeIndexParams(4));

		server.run();
	}
	catch (std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}

