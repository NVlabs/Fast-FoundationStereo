docker rm -f ffs || true
xhost +local:root || true
DIR=$(pwd)/../
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name ffs --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:/workspace --ipc=host -e DISPLAY=${DISPLAY} -e LIBGL_ALWAYS_SOFTWARE=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp -v /home:/home -v /mnt:/mnt ffs bash
