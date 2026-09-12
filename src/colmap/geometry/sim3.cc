// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/sim3.h"

#include "colmap/util/logging.h"

#include <fstream>
#include <locale>

namespace colmap {

void Sim3d::ToFile(const std::filesystem::path& path) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK(file.good()) << path;
  file.imbue(std::locale::classic());
  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);
  file << scale() << " " << rotation().w() << " " << rotation().x() << " "
       << rotation().y() << " " << rotation().z() << " " << translation().x()
       << " " << translation().y() << " " << translation().z() << '\n';
}

Sim3d Sim3d::FromFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK(file.good()) << path;
  file.imbue(std::locale::classic());
  Sim3d t;
  THROW_CHECK(file >> t.scale() >> t.rotation().w() >> t.rotation().x() >>
              t.rotation().y() >> t.rotation().z() >> t.translation().x() >>
              t.translation().y() >> t.translation().z());
  return t;
}

std::ostream& operator<<(std::ostream& stream, const Sim3d& tform) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "Sim3d(scale=" << tform.scale() << ", rotation_xyzw=["
         << tform.rotation().coeffs().format(kVecFmt) << "], translation=["
         << tform.translation().format(kVecFmt) << "])";
  return stream;
}

}  // namespace colmap
