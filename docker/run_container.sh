docker rm -f ffs
DIR=$(pwd)/../
mkdir -p /tmp/runtime
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name ffs --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:/workspace --ipc=host -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 -e QT_DEBUG_PLUGINS=0 -e XDG_RUNTIME_DIR=/tmp/runtime -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp -v /home:/home -v /mnt:/mnt ffs bash