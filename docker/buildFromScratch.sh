docker build -t="colmap-ubuntu" .;
docker run -w /working -v $1:/working -it colmap-ubuntu;