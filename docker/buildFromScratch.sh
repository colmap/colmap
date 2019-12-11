docker build -t="colmap-ubuntu" .;
docker run --gpus all -w /working -v $1:/working -it colmap-ubuntu;