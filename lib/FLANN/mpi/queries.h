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


#ifndef MPI_QUERIES_H_
#define MPI_QUERIES_H_

#include <FLANN/mpi/matrix.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/asio.hpp>

namespace flann
{

template<typename T>
struct Request
{
	flann::Matrix<T> queries;
	int nn;
	int checks;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & queries & nn & checks;
	}
};

template<typename T>
struct Response
{
	flann::Matrix<int> indices;
	flann::Matrix<T> dists;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & indices & dists;
	}
};


using boost::asio::ip::tcp;

template <typename T>
void read_object(tcp::socket& sock, T& val)
{
	uint32_t size;
	boost::asio::read(sock, boost::asio::buffer(&size, sizeof(size)));
	size = ntohl(size);

	boost::asio::streambuf archive_stream;
	boost::asio::read(sock, archive_stream, boost::asio::transfer_at_least(size));

	boost::archive::binary_iarchive archive(archive_stream);
	archive >> val;
}

template <typename T>
void write_object(tcp::socket& sock, const T& val)
{
	boost::asio::streambuf archive_stream;
	boost::archive::binary_oarchive archive(archive_stream);
	archive << val;

	uint32_t size = archive_stream.size();
	size = htonl(size);
	boost::asio::write(sock, boost::asio::buffer(&size, sizeof(size)));
	boost::asio::write(sock, archive_stream);

}

}



#endif /* MPI_QUERIES_H_ */
