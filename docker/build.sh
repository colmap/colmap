docker build -t="colmap:latest" .;
docker run --gpus all -w /working -v $1:/working -it colmap:latest;
