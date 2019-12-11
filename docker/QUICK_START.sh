docker pull colmap/colmap:3.6-dev.3
docker run -w /working -v $1:/working -it colmap/colmap:3.6-dev.3;