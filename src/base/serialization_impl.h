#ifndef COLMAP_SRC_BASE_SERIALIZATION_IMPL_H_
#define COLMAP_SRC_BASE_SERIALIZATION_IMPL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "ext/yas/yas.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "util/misc.h"

namespace colmap {

namespace internal {
template <typename Type, typename OutputStream>
void WriteToBinaryStream(OutputStream&& output_stream, const Type& data) {
  yas::binary_oarchive<OutputStream> output_archive(output_stream);
  output_archive & data;
}

template <typename Type, typename InputStream>
void ReadFromBinaryStream(InputStream&& input_stream, Type* data) {
  CHECK_NOTNULL(data);
  yas::binary_iarchive<InputStream> input_archive(input_stream);
  input_archive & *data;
}
}  // namespace internal

template <typename Type>
void WriteToBinaryFile(const std::string& path, const Type& data) {
  internal::WriteToBinaryStream(
      yas::file_ostream(path.c_str(), yas::file_mode::file_trunc), data);
}

template <typename Type>
void ReadFromBinaryFile(const std::string& path, Type* data) {
  internal::ReadFromBinaryStream(yas::file_istream(path.c_str()), data);
}

// The only purpose of this class is to provide a clean way to access private
// and proteceted members of classes we are interested in serializing. To use
// it, simply add this class as a friend of the class to serialize. Afterwards,
// one can easily write bidirectional serialization methods (i.e. the same
// method serializes and deserializes) that specialize for each befriended
// class.
struct Serializator {
#define DECLARE_SERIALIZATION_TYPE(TYPE)           \
  template <typename Archive>                      \
  static void Apply(Archive& archive, TYPE& type);
DECLARE_SERIALIZATION_TYPE(Point2D)
DECLARE_SERIALIZATION_TYPE(Point3D)
DECLARE_SERIALIZATION_TYPE(TrackElement)
DECLARE_SERIALIZATION_TYPE(Track)
DECLARE_SERIALIZATION_TYPE(Camera)
DECLARE_SERIALIZATION_TYPE(Image)
DECLARE_SERIALIZATION_TYPE(VisibilityPyramid)
DECLARE_SERIALIZATION_TYPE(SceneGraph::Correspondence)
DECLARE_SERIALIZATION_TYPE(SceneGraph::Image)
DECLARE_SERIALIZATION_TYPE(SceneGraph)
DECLARE_SERIALIZATION_TYPE(Reconstruction)
#undef DECLARE_SERIALIZATION_TYPE

  // Typedef the following private Structure for serialization.
  typedef SceneGraph::Image SceneGraphImage;
};

// The serialize<>() functions that YAS expects for each type. They use the
// Serializator since this object has direct acess to the members of the objects
// to be serialized by being friend to each respective class.
#define REGISTER_SERIALIZATION_TYPE(TYPE)        \
  template <typename Archive>                    \
  void serialize(Archive& archive, TYPE& data) { \
    Serializator::Apply(archive, data);          \
  }
REGISTER_SERIALIZATION_TYPE(Point2D)
REGISTER_SERIALIZATION_TYPE(Point3D)
REGISTER_SERIALIZATION_TYPE(TrackElement)
REGISTER_SERIALIZATION_TYPE(Track)
REGISTER_SERIALIZATION_TYPE(Camera)
REGISTER_SERIALIZATION_TYPE(Image)
REGISTER_SERIALIZATION_TYPE(VisibilityPyramid)
REGISTER_SERIALIZATION_TYPE(SceneGraph::Correspondence)
REGISTER_SERIALIZATION_TYPE(Serializator::SceneGraphImage)
REGISTER_SERIALIZATION_TYPE(SceneGraph)
REGISTER_SERIALIZATION_TYPE(Reconstruction)
#undef REGISTER_SERIALIZATION_TYPE

// -------------------------------------------------------------------------- //
// ------------------- Implementation of Serializator ----------------------- //
// -------------------------------------------------------------------------- //
template <typename Archive>
void Serializator::Apply(Archive& archive, Point2D& point) {
  archive & point.xy_ & point.point3D_id_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, TrackElement& element) {
  archive & element.image_id & element.point2D_idx;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, Track& track) {
  archive & track.elements_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, Point3D& point) {
  archive & point.xyz_ & point.color_ & point.error_ & point.track_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, Camera& camera) {
  archive & camera.camera_id_ & camera.model_id_ & camera.width_
          & camera.height_ & camera.params_ & camera.prior_focal_length_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, VisibilityPyramid& pyramid) {
  archive & pyramid.width_ & pyramid.height_ & pyramid.score_
          & pyramid.max_score_ & pyramid.pyramid_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, Image& image) {
  archive & image.image_id_ & image.name_ & image.camera_id_
          & image.registered_ & image.num_points3D_ & image.num_observations_
          & image.num_correspondences_ & image.num_visible_points3D_
          & image.qvec_ & image.tvec_ & image.qvec_prior_ & image.tvec_prior_
          & image.points2D_ & image.num_correspondences_have_point3D_
          & image.point3D_visibility_pyramid_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive,
                  SceneGraph::Correspondence& correspondence) {
  archive & correspondence.image_id
          & correspondence.point2D_idx;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, SceneGraphImage& scene_graph_image) {
  archive & scene_graph_image.num_observations
          & scene_graph_image.num_correspondences
          & scene_graph_image.corrs;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, SceneGraph& scene_graph) {
  archive & scene_graph.images_ & scene_graph.image_pairs_;
}

template <typename Archive>
void Serializator::Apply(Archive& archive, Reconstruction& reconstruction) {
  archive & reconstruction.cameras_ & reconstruction.images_
          & reconstruction.points3D_ & reconstruction.image_pairs_ &
          reconstruction.reg_image_ids_ & reconstruction.num_added_points3D_;
}

}  // namespace colmap

// -------------------------------------------------------------------------- //
// -------------------- External Objects Serialization ---------------------- //
// -------------------------------------------------------------------------- //
// These methods handle the serialization and deserialization of Eigen matrices.
// The functions are very general and can handle expressions as input or output,
// this is useful in case we want to store to disk only a portion of a matrix.
// Similarly, one can read back from file data to only a region of a matrix
// (using block() methods) provided enough memory has been allocated for the
// underlying matrix. See serialization_tests.cc for examples.
namespace Eigen {
template <typename Archive, typename Derived>
void serialize(Archive& archive, const MatrixBase<Derived>& matrix) {
  matrix.derived().eval();
  archive & matrix.rows() & matrix.cols();
  for (int c = 0; c < matrix.cols(); ++c) {
    for (int r = 0; r < matrix.rows(); ++r) {
      archive & matrix(r, c);
    }
  }
}

template <typename Archive, typename Derived>
void serialize(Archive& archive, MatrixBase<Derived>& matrix) {
  typename MatrixBase<Derived>::Index rows_in_file, cols_in_file;
  archive & rows_in_file & cols_in_file;
  // Depending on the type of input matrix and read data, make the appropriate
  // checks.
  if (MatrixBase<Derived>::ColsAtCompileTime == Dynamic) {
    if (MatrixBase<Derived>::RowsAtCompileTime == Dynamic) {
      // Fully dynamic matrix, e.g. Eigen::MatrixXd.
    } else {
      // E.g. Eigen::Matrix<int, 4, Eigen::Dynamic>.
      CHECK_EQ(rows_in_file, MatrixBase<Derived>::RowsAtCompileTime);
    }
  } else {
    CHECK_EQ(cols_in_file, MatrixBase<Derived>::ColsAtCompileTime);
    if (MatrixBase<Derived>::RowsAtCompileTime == Dynamic) {
      // E.g. Eigen::Matrix<double, Eigen::Dynamic, 12>.
    } else {
      // Fully static, e.g. Eigen::Matrix3d or Eigen::Vector3d.
      CHECK_EQ(rows_in_file, MatrixBase<Derived>::RowsAtCompileTime);
    }
  }

  matrix.derived().resize(rows_in_file, cols_in_file);
  for (int c = 0; c < matrix.cols(); ++c) {
    for (int r = 0; r < matrix.rows(); ++r) {
      archive & matrix(r, c);
    }
  }
}
}  // namespace Eigen

// -------------------------------------------------------------------------- //
// YAS provides functionality to serialize virtually all containers from the
// STL. However, it does not support custom allocators for them.  The following
// code specializes the serialization and deserialization of objects of type
// std::unordered_map with Eigen::aligned_allocator as allocator.
namespace yas {
namespace detail {
template <std::size_t F, typename K, typename V>
struct serializer<
    type_prop::not_a_fundamental, ser_method::use_internal_serializer, F,
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       Eigen::aligned_allocator<std::pair<const K, V>>>> {
  typedef typename std::unordered_map<
      K, V, std::hash<K>, std::equal_to<K>,
      Eigen::aligned_allocator<std::pair<const K, V>>> MapType;
  template <typename Archive>
  static Archive& save(Archive& ar, const MapType& map) {
    ar.write_seq_size(map.size());
    for (const auto& it : map) {
      ar & it.first & it.second;
    }
    return ar;
  }

  template <typename Archive>
  static Archive& load(Archive& ar, MapType& map) {
    auto size = ar.read_seq_size();
    for (; size; --size) {
      K key{};
      V val{};
      ar & key & val;
      map.insert(std::make_pair(std::move(key), std::move(val)));
    }
    return ar;
  }
};
}  // namespace detail
}  // namespace yas

#endif  // COLMAP_SRC_BASE_SERIALIZATION_IMPL_H_
