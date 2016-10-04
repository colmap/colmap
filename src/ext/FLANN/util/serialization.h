#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include "ext/FLANN/ext/lz4.h"
#include "ext/FLANN/ext/lz4hc.h"


namespace flann
{
    struct IndexHeaderStruct {
        char signature[24];
        char version[16];
        flann_datatype_t data_type;
        flann_algorithm_t index_type;
        size_t rows;
        size_t cols;
        size_t compression;
        size_t first_block_size;
    };

namespace serialization
{

struct access
{
    template<typename Archive, typename T>
    static inline void serialize(Archive& ar, T& type)
    {
        type.serialize(ar);
    }
};


template<typename Archive, typename T>
inline void serialize(Archive& ar, T& type)
{
    access::serialize(ar,type);
}

template<typename T>
struct Serializer
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T& val)
    {
        serialization::serialize(ar,val);
    }
    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const T& val)
    {
        serialization::serialize(ar,const_cast<T&>(val));
    }
};

#define BASIC_TYPE_SERIALIZER(type)\
template<> \
struct Serializer<type> \
{\
    template<typename InputArchive>\
    static inline void load(InputArchive& ar, type& val)\
    {\
        ar.load(val);\
    }\
    template<typename OutputArchive>\
    static inline void save(OutputArchive& ar, const type& val)\
    {\
        ar.save(val);\
    }\
}

#define ENUM_SERIALIZER(type)\
template<>\
struct Serializer<type>\
{\
    template<typename InputArchive>\
    static inline void load(InputArchive& ar, type& val)\
    {\
        int int_val;\
        ar & int_val;\
        val = (type) int_val;\
    }\
    template<typename OutputArchive>\
    static inline void save(OutputArchive& ar, const type& val)\
    {\
        int int_val = (int)val;\
        ar & int_val;\
    }\
}


// declare serializers for simple types
BASIC_TYPE_SERIALIZER(char);
BASIC_TYPE_SERIALIZER(unsigned char);
BASIC_TYPE_SERIALIZER(short);
BASIC_TYPE_SERIALIZER(unsigned short);
BASIC_TYPE_SERIALIZER(int);
BASIC_TYPE_SERIALIZER(unsigned int);
BASIC_TYPE_SERIALIZER(long);
BASIC_TYPE_SERIALIZER(unsigned long);
BASIC_TYPE_SERIALIZER(unsigned long long);
BASIC_TYPE_SERIALIZER(float);
BASIC_TYPE_SERIALIZER(double);
BASIC_TYPE_SERIALIZER(bool);
#ifdef _MSC_VER
// unsigned __int64 ~= unsigned long long
// Will throw error on VS2013
#if _MSC_VER < 1800
BASIC_TYPE_SERIALIZER(unsigned __int64);
#endif
#endif


// serializer for std::vector
template<typename T>
struct Serializer<std::vector<T> >
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, std::vector<T>& val)
    {
        size_t size;
        ar & size;
        val.resize(size);
        for (size_t i=0;i<size;++i) {
            ar & val[i];
        }
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const std::vector<T>& val)
    {
        ar & val.size();
        for (size_t i=0;i<val.size();++i) {
            ar & val[i];
        }
    }
};

// serializer for std::vector
template<typename K, typename V>
struct Serializer<std::map<K,V> >
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, std::map<K,V>& map_val)
    {
        size_t size;
        ar & size;
        for (size_t i = 0; i < size; ++i)
        {
            K key;
            ar & key;
            V value;
            ar & value;
            map_val[key] = value;
        }
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const std::map<K,V>& map_val)
    {
        ar & map_val.size();
        for (typename std::map<K,V>::const_iterator i=map_val.begin(); i!=map_val.end(); ++i) {
            ar & i->first;
            ar & i->second;
        }
    }
};

template<typename T>
struct Serializer<T*>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T*& val)
    {
        ar.load(val);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  T* const& val)
    {
        ar.save(val);
    }
};

template<typename T, int N>
struct Serializer<T[N]>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T (&val)[N])
    {
        ar.load(val);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  T const (&val)[N])
    {
        ar.save(val);
    }
};




struct binary_object
{
    void const * ptr_;
    size_t size_;

    binary_object( void * const ptr, size_t size) :
        ptr_(ptr),
        size_(size)
    {}
    binary_object(const binary_object & rhs) :
        ptr_(rhs.ptr_),
        size_(rhs.size_)
    {}

    binary_object & operator=(const binary_object & rhs) {
        ptr_ = rhs.ptr_;
        size_ = rhs.size_;
        return *this;
    }
};

inline const binary_object make_binary_object(/* const */ void * t, size_t size){
    return binary_object(t, size);
}

template<>
struct Serializer<const binary_object>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, const binary_object& b)
    {
        ar.load_binary(const_cast<void *>(b.ptr_), b.size_);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  const binary_object& b)
    {
        ar.save_binary(b.ptr_, b.size_);
    }
};

template<>
struct Serializer<binary_object>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, binary_object& b)
    {
        ar.load_binary(const_cast<void *>(b.ptr_), b.size_);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  const binary_object& b)
    {
        ar.save_binary(b.ptr_, b.size_);
    }
};



template <bool C_>
struct bool_ {
    static const bool value = C_;
    typedef bool value_type;
};


class ArchiveBase
{
public:
	void* getObject() { return object_; }

	void setObject(void* object) { object_ = object; }

private:
	void* object_;
};


template<typename Archive>
class InputArchive : public ArchiveBase
{
protected:
    InputArchive() {};
public:
    typedef bool_<true> is_loading;
    typedef bool_<false> is_saving;

    template<typename T>
    Archive& operator& (T& val)
    {
        Serializer<T>::load(*static_cast<Archive*>(this),val);
        return *static_cast<Archive*>(this);
    }
};


template<typename Archive>
class OutputArchive : public ArchiveBase
{
protected:
    OutputArchive() {};
public:
    typedef bool_<false> is_loading;
    typedef bool_<true> is_saving;

    template<typename T>
    Archive& operator& (const T& val)
    {
        Serializer<T>::save(*static_cast<Archive*>(this),val);
        return *static_cast<Archive*>(this);
    }
};



class SizeArchive : public OutputArchive<SizeArchive>
{
    size_t size_;
public:

    SizeArchive() : size_(0)
    {
    }

    template<typename T>
    void save(const T& val)
    {
        size_ += sizeof(val);
    }

    template<typename T>
    void save_binary(T* ptr, size_t size)
    {
    	size_ += size;
    }


    void reset()
    {
        size_ = 0;
    }

    size_t size()
    {
        return size_;
    }
};


//
//class PrintArchive : public OutputArchive<PrintArchive>
//{
//public:
//    template<typename T>
//    void save(const T& val)
//    {
//        std::cout << val << std::endl;
//    }
//
//    template<typename T>
//    void save_binary(T* ptr, size_t size)
//    {
//        std::cout << "<binary object>" << std::endl;
//    }
//};

#define BLOCK_BYTES (1024 * 64)

class SaveArchive : public OutputArchive<SaveArchive>
{
    /**
     * Based on blockStreaming_doubleBuffer code at:
     * https://github.com/Cyan4973/lz4/blob/master/examples/blockStreaming_doubleBuffer.c
     */

    FILE* stream_;
    bool own_stream_;
    char *buffer_;
    size_t offset_;

    int first_block_;
    char *buffer_blocks_;
    char *compressed_buffer_;
    LZ4_streamHC_t lz4Stream_body;
    LZ4_streamHC_t* lz4Stream;

    void initBlock()
    {
        // Alloc the space for both buffer blocks (each compressed block
        // references the previous)
        buffer_ = buffer_blocks_ = (char *)malloc(BLOCK_BYTES*2);
        compressed_buffer_ = (char *)malloc(LZ4_COMPRESSBOUND(BLOCK_BYTES) + sizeof(size_t));
        if (buffer_ == NULL || compressed_buffer_ == NULL) {
            throw FLANNException("Error allocating compression buffer");
        }

        // Init the LZ4 stream
        lz4Stream = &lz4Stream_body;
        LZ4_resetStreamHC(lz4Stream, 9);
        first_block_ = true;

        offset_ = 0;
    }

    void flushBlock()
    {
        size_t compSz = 0;
        // Handle header
        if (first_block_) {
            // Copy & set the header
            IndexHeaderStruct *head = (IndexHeaderStruct *)buffer_;
            size_t headSz = sizeof(IndexHeaderStruct);

            assert(head->compression == 0);
            head->compression = 1; // Bool now, enum later

            // Do the compression for the block
            compSz = LZ4_compress_HC_continue(
                lz4Stream, buffer_+headSz, compressed_buffer_+headSz, offset_-headSz,
                LZ4_COMPRESSBOUND(BLOCK_BYTES));

            if(compSz <= 0) {
                throw FLANNException("Error compressing (first block)");
            }

            // Handle header
            head->first_block_size = compSz;
            memcpy(compressed_buffer_, buffer_, headSz);

            compSz += headSz;
            first_block_ = false;
        } else {
            size_t headSz = sizeof(compSz);

            // Do the compression for the block
            compSz = LZ4_compress_HC_continue(
                lz4Stream, buffer_, compressed_buffer_+headSz, offset_,
                LZ4_COMPRESSBOUND(BLOCK_BYTES));

            if(compSz <= 0) {
                throw FLANNException("Error compressing");
            }

            // Save the size of the compressed block as the header
            memcpy(compressed_buffer_, &compSz, headSz);
            compSz += headSz;
        }

        // Write the compressed buffer
        fwrite(compressed_buffer_, compSz, 1, stream_);

        // Switch the buffer to the *other* block
        if (buffer_ == buffer_blocks_)
            buffer_ = &buffer_blocks_[BLOCK_BYTES];
        else
            buffer_ = buffer_blocks_;
        offset_ = 0;
    }

    void endBlock()
    {
        // Cleanup memory
        free(buffer_blocks_);
        buffer_blocks_ = NULL;
        buffer_ = NULL;
        free(compressed_buffer_);
        compressed_buffer_ = NULL;

        // Write a '0' size for next block
        size_t z = 0;
        fwrite(&z, sizeof(z), 1, stream_);
    }

public:
    SaveArchive(const char* filename)
    {
        stream_ = fopen(filename, "wb");
        own_stream_ = true;
        initBlock();
    }

    SaveArchive(FILE* stream) : stream_(stream), own_stream_(false)
    {
        initBlock();
    }

    ~SaveArchive()
    {
        flushBlock();
        endBlock();
        if (buffer_) {
            free(buffer_);
            buffer_ = NULL;
        }
    	if (own_stream_) {
    		fclose(stream_);
    	}
    }

    template<typename T>
    void save(const T& val)
    {
        assert(sizeof(val) < BLOCK_BYTES);
        if (offset_+sizeof(val) > BLOCK_BYTES)
            flushBlock();
        memcpy(buffer_+offset_, &val, sizeof(val));
        offset_ += sizeof(val);
    }

    template<typename T>
    void save(T* const& val)
    {
    	// don't save pointers
        //fwrite(&val, sizeof(val), 1, handle_);
    }

    template<typename T>
    void save_binary(T* ptr, size_t size)
    {
        while (size > BLOCK_BYTES) {
            // Flush existing block
            flushBlock();

            // Save large chunk
            memcpy(buffer_, ptr, BLOCK_BYTES);
            offset_ += BLOCK_BYTES;
            ptr = ((char *)ptr) + BLOCK_BYTES;
            size -= BLOCK_BYTES;
        }

        // Save existing block if new data will make it too big
        if (offset_+size > BLOCK_BYTES)
            flushBlock();

        // Copy out requested data
        memcpy(buffer_+offset_, ptr, size);
        offset_ += size;
    }

};


class LoadArchive : public InputArchive<LoadArchive>
{
    /**
     * Based on blockStreaming_doubleBuffer code at:
     * https://github.com/Cyan4973/lz4/blob/master/examples/blockStreaming_doubleBuffer.c
     */

    FILE* stream_;
    bool own_stream_;
    char *buffer_;
    char *ptr_;

    char *buffer_blocks_;
    char *compressed_buffer_;
    LZ4_streamDecode_t lz4StreamDecode_body;
    LZ4_streamDecode_t* lz4StreamDecode;
    size_t block_sz_;

    void decompressAndLoadV10(FILE* stream)
    {
        buffer_ = NULL;

        // Find file size
        size_t pos = ftell(stream);
        fseek(stream, 0, SEEK_END);
        size_t fileSize = ftell(stream)-pos;
        fseek(stream, pos, SEEK_SET);
        size_t headSz = sizeof(IndexHeaderStruct);

        // Read the (compressed) file to a buffer
        char *compBuffer = (char *)malloc(fileSize);
        if (compBuffer == NULL) {
            throw FLANNException("Error allocating file buffer space");
        }
        if (fread(compBuffer, fileSize, 1, stream) != 1) {
            free(compBuffer);
            throw FLANNException("Invalid index file, cannot read from disk (compressed)");
        }

        // Extract header
        IndexHeaderStruct *head = (IndexHeaderStruct *)(compBuffer);

        // Backward compatability
        size_t compressedSz = fileSize-headSz;
        size_t uncompressedSz = head->first_block_size-headSz;

        // Check for compression type
        if (head->compression != 1) {
            free(compBuffer);
            throw FLANNException("Compression type not supported");
        }

        // Allocate a decompressed buffer
        ptr_ = buffer_ = (char *)malloc(uncompressedSz+headSz);
        if (buffer_ == NULL) {
            free(compBuffer);
            throw FLANNException("Error (re)allocating decompression buffer");
        }

        // Extract body
        size_t usedSz = LZ4_decompress_safe(compBuffer+headSz,
                                            buffer_+headSz,
                                            compressedSz,
                                            uncompressedSz);

        // Check if the decompression was the expected size.
        if (usedSz != uncompressedSz) {
            free(compBuffer);
            throw FLANNException("Unexpected decompression size");
        }

        // Copy header data
        memcpy(buffer_, compBuffer, headSz);
        free(compBuffer);

        // Put the file pointer at the end of the data we've read
        if (compressedSz+headSz+pos != fileSize)
            fseek(stream, compressedSz+headSz+pos, SEEK_SET);
        block_sz_ = uncompressedSz+headSz;
    }

    void initBlock(FILE *stream)
    {
        size_t pos = ftell(stream);
        buffer_ = NULL;
        buffer_blocks_ = NULL;
        compressed_buffer_ = NULL;
        size_t headSz = sizeof(IndexHeaderStruct);

        // Read the file header to a buffer
        IndexHeaderStruct *head = (IndexHeaderStruct *)malloc(headSz);
        if (head == NULL) {
            throw FLANNException("Error allocating header buffer space");
        }
        if (fread(head, headSz, 1, stream) != 1) {
            free(head);
            throw FLANNException("Invalid index file, cannot read from disk (header)");
        }

        // Backward compatability
        if (head->signature[13] == '1' && head->signature[15] == '0') {
            free(head);
            fseek(stream, pos, SEEK_SET);
            return decompressAndLoadV10(stream);
        }

        // Alloc the space for both buffer blocks (each block
        // references the previous)
        buffer_ = buffer_blocks_ = (char *)malloc(BLOCK_BYTES*2);
        compressed_buffer_ = (char *)malloc(LZ4_COMPRESSBOUND(BLOCK_BYTES));
        if (buffer_ == NULL || compressed_buffer_ == NULL) {
            free(head);
            throw FLANNException("Error allocating compression buffer");
        }

        // Init the LZ4 stream
        lz4StreamDecode = &lz4StreamDecode_body;
        LZ4_setStreamDecode(lz4StreamDecode, NULL, 0);

        // Read first block
        memcpy(buffer_, head, headSz);
        loadBlock(buffer_+headSz, head->first_block_size, stream);
        block_sz_ += headSz;
        ptr_ = buffer_;
        free(head);
    }

    void loadBlock(char* buffer_, size_t compSz, FILE* stream)
    {
        if(compSz >= LZ4_COMPRESSBOUND(BLOCK_BYTES)) {
            throw FLANNException("Requested block size too large");
        }

        // Read the block into the compressed buffer
        if (fread(compressed_buffer_, compSz, 1, stream) != 1) {
            throw FLANNException("Invalid index file, cannot read from disk (block)");
        }

        // Decompress into the regular buffer
        const int decBytes = LZ4_decompress_safe_continue(
            lz4StreamDecode, compressed_buffer_, buffer_, compSz, BLOCK_BYTES);
        if(decBytes <= 0) {
            throw FLANNException("Invalid index file, cannot decompress block");
        }
        block_sz_ = decBytes;
    }

    void preparePtr(size_t size)
    {
        // Return if the new size is less than (or eq) the size of a block
        if (ptr_+size <= buffer_+block_sz_)
            return;

        // Switch the buffer to the *other* block
        if (buffer_ == buffer_blocks_)
            buffer_ = &buffer_blocks_[BLOCK_BYTES];
        else
            buffer_ = buffer_blocks_;

        // Find the size of the next block
        size_t cmpSz = 0;
        size_t readCnt = fread(&cmpSz, sizeof(cmpSz), 1, stream_);
        if(cmpSz <= 0 || readCnt != 1) {
            throw FLANNException("Requested to read next block past end of file");
        }

        // Load block & init ptr
        loadBlock(buffer_, cmpSz, stream_);
        ptr_ = buffer_;
    }

    void endBlock()
    {
        // If not v1.0 format hack...
        if (buffer_blocks_ != NULL) {
            // Read the last '0' in the file
            size_t zero = -1;
            if (fread(&zero, sizeof(zero), 1, stream_) != 1) {
                throw FLANNException("Invalid index file, cannot read from disk (end)");
            }
            if (zero != 0) {
                throw FLANNException("Invalid index file, last block not zero length");
            }
        }

        // Free resources
        if (buffer_blocks_ != NULL) {
            free(buffer_blocks_);
            buffer_blocks_ = NULL;
        }
        if (compressed_buffer_ != NULL) {
            free(compressed_buffer_);
            compressed_buffer_ = NULL;
        }
        ptr_ = NULL;
    }

public:
    LoadArchive(const char* filename)
    {
        // Open the file
        stream_ = fopen(filename, "rb");
        own_stream_ = true;

        initBlock(stream_);
    }

    LoadArchive(FILE* stream)
    {
        stream_ = stream;
        own_stream_ = false;

        initBlock(stream);
    }

    ~LoadArchive()
    {
        endBlock();
    	if (own_stream_) {
    		fclose(stream_);
    	}
    }

    template<typename T>
    void load(T& val)
    {
        preparePtr(sizeof(val));
        memcpy(&val, ptr_, sizeof(val));
        ptr_ += sizeof(val);
    }

    template<typename T>
    void load(T*& val)
    {
    	// don't load pointers
        //fread(&val, sizeof(val), 1, handle_);
    }

    template<typename T>
    void load_binary(T* ptr, size_t size)
    {
        while (size > BLOCK_BYTES) {
            // Load next block
            preparePtr(BLOCK_BYTES);

            // Load large chunk
            memcpy(ptr, ptr_, BLOCK_BYTES);
            ptr_ += BLOCK_BYTES;
            ptr = ((char *)ptr) + BLOCK_BYTES;
            size -= BLOCK_BYTES;
        }

        // Load next block if needed
        preparePtr(size);

        // Load the data
        memcpy(ptr, ptr_, size);
        ptr_ += size;
    }
};

} // namespace serialization
} // namespace flann
#endif // SERIALIZATION_H_
