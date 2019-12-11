docker build -t="colmap:3.6-dev.3" .;
docker run --gpus all -w /working -v $1:/working -it colmap:3.6-dev.3;