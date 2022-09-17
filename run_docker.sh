docker run --rm -i \
  -p 8080:8000 \
  -p 8081:8001 \
  -p 8082:8002 \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
  -v ${MODEL_REPOSITORY_PATH}:${MODEL_REPOSITORY_PATH} \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tritonserver:22.01-py3 
