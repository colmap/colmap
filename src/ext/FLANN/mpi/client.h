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


#ifndef MPI_CLIENT_H_
#define MPI_CLIENT_H_

#include <cstdlib>
#include <boost/asio.hpp>
#include <ext/FLANN/util/matrix.h>
#include <ext/FLANN/util/params.h>
#include "queries.h"

namespace flann {
namespace mpi {


class Client
{
public:
	Client(const std::string& host, const std::string& service)
	{
	    tcp::resolver resolver(io_service_);
	    tcp::resolver::query query(tcp::v4(), host, service);
	    iterator_ = resolver.resolve(query);
	}


	template<typename ElementType, typename DistanceType>
	void knnSearch(const flann::Matrix<ElementType>& queries, flann::Matrix<int>& indices, flann::Matrix<DistanceType>& dists, int knn, const SearchParams& params)
	{
	    tcp::socket sock(io_service_);
	    sock.connect(*iterator_);

	    Request<ElementType> req;
	    req.nn = knn;
	    req.queries = queries;
	    req.checks = params.checks;
	    // send request
	    write_object(sock,req);

	    Response<DistanceType> resp;
	    // read response
	    read_object(sock, resp);

	    for (size_t i=0;i<indices.rows;++i) {
	    	for (size_t j=0;j<indices.cols;++j) {
	    		indices[i][j] = resp.indices[i][j];
	    		dists[i][j] = resp.dists[i][j];
	    	}
	    }
	}


private:
	boost::asio::io_service io_service_;
	tcp::resolver::iterator iterator_;
};


} //namespace mpi
} // namespace flann

#endif // MPI_CLIENT_H_
