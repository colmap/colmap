/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE NNIndexGOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_SAVING_H_
#define FLANN_SAVING_H_

#include <cstring>
#include <vector>
#include <stdio.h>

#include "FLANN/general.h"
#include "FLANN/util/serialization.h"


#ifdef FLANN_SIGNATURE_
#undef FLANN_SIGNATURE_
#endif
#define FLANN_SIGNATURE_ "FLANN_INDEX_v1.1"

namespace flann
{

/**
 * Structure representing the index header.
 */
struct IndexHeader
{
    IndexHeaderStruct h;

    IndexHeader()
	{
        memset(h.signature, 0, sizeof(h.signature));
        strcpy(h.signature, FLANN_SIGNATURE_);
        memset(h.version, 0, sizeof(h.version));
        strcpy(h.version, FLANN_VERSION_);

        h.compression = 0;
        h.first_block_size = 0;
	}

private:
    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar & h.signature;
        ar & h.version;
        ar & h.data_type;
        ar & h.index_type;
        ar & h.rows;
        ar & h.cols;
        ar & h.compression;
        ar & h.first_block_size;
    }
    friend struct serialization::access;
};

/**
 * Saves index header to stream
 *
 * @param stream - Stream to save to
 * @param index - The index to save
 */
template<typename Index>
void save_header(FILE* stream, const Index& index)
{
    IndexHeader header;
    header.h.data_type = flann_datatype_value<typename Index::ElementType>::value;
    header.h.index_type = index.getType();
    header.h.rows = index.size();
    header.h.cols = index.veclen();

    fwrite(&header, sizeof(header),1,stream);
}


/**
 *
 * @param stream - Stream to load from
 * @return Index header
 */
inline IndexHeader load_header(FILE* stream)
{
    IndexHeader header;
    int read_size = fread(&header,sizeof(header),1,stream);

    if (read_size != 1) {
        throw FLANNException("Invalid index file, cannot read");
    }

    if (strncmp(header.h.signature,
                FLANN_SIGNATURE_,
                strlen(FLANN_SIGNATURE_) - strlen("v0.0")) != 0) {
        throw FLANNException("Invalid index file, wrong signature");
    }

    return header;
}


namespace serialization
{
ENUM_SERIALIZER(flann_algorithm_t);
ENUM_SERIALIZER(flann_centers_init_t);
ENUM_SERIALIZER(flann_log_level_t);
ENUM_SERIALIZER(flann_datatype_t);
}

}

#endif /* FLANN_SAVING_H_ */
