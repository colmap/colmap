/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifndef MPI_SERVER_H_
#define MPI_SERVER_H_

#include <FLANN/mpi/index.h>
#include <stdio.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

#include "queries.h"

namespace flann {

namespace mpi {

template<typename Distance>
class Server
{

	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;
	typedef boost::shared_ptr<tcp::socket> socket_ptr;
	typedef flann::mpi::Index<Distance> FlannIndex;

	void session(socket_ptr sock)
	{
		boost::mpi::communicator world;
		try {
			Request<ElementType> req;
			if (world.rank()==0) {
				read_object(*sock,req);
				std::cout << "Received query\n";
			}
			// broadcast request to all MPI processes
			boost::mpi::broadcast(world, req, 0);

			Response<DistanceType> resp;
			if (world.rank()==0) {
				int rows = req.queries.rows;
				int cols = req.nn;
				resp.indices = flann::Matrix<int>(new int[rows*cols], rows, cols);
				resp.dists = flann::Matrix<DistanceType>(new DistanceType[rows*cols], rows, cols);
			}

			std::cout << "Searching in process " << world.rank() << "\n";
			index_->knnSearch(req.queries, resp.indices, resp.dists, req.nn, flann::SearchParams(req.checks));

			if (world.rank()==0) {
				std::cout << "Sending result\n";
				write_object(*sock,resp);
			}

			delete[] req.queries.ptr();
			if (world.rank()==0) {
				delete[] resp.indices.ptr();
				delete[] resp.dists.ptr();
			}

		}
		catch (std::exception& e) {
			std::cerr << "Exception in thread: " << e.what() << "\n";
		}
	}



public:
	Server(const std::string& filename, const std::string& dataset, short port, const IndexParams& params) :
		port_(port)
	{
		boost::mpi::communicator world;
		if (world.rank()==0) {
			std::cout << "Reading dataset and building index...";
			std::flush(std::cout);
		}
		index_ = new FlannIndex(filename, dataset, params);
		index_->buildIndex();
		world.barrier(); // wait for data to be loaded and indexes to be created
		if (world.rank()==0) {
			std::cout << "done.\n";
		}
	}


	void run()
	{
		boost::mpi::communicator world;
		boost::shared_ptr<boost::asio::io_service> io_service;
		boost::shared_ptr<tcp::acceptor> acceptor;

		if (world.rank()==0) {
			io_service.reset(new boost::asio::io_service());
			acceptor.reset(new tcp::acceptor(*io_service, tcp::endpoint(tcp::v4(), port_)));
			std::cout << "Start listening for queries...\n";
		}
		for (;;) {
			socket_ptr sock;
			if (world.rank()==0) {
				sock.reset(new tcp::socket(*io_service));
				acceptor->accept(*sock);
				std::cout << "Accepted connection\n";
			}
			world.barrier(); // everybody waits here for a connection
			boost::thread t(boost::bind(&Server::session, this, sock));
			t.join();
		}

	}

private:
	FlannIndex* index_;
	short port_;
};


} // namespace mpi
} // namespace flann

#endif // MPI_SERVER_H_
