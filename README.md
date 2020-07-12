# NNIE Transfer Server
This repo enables you to load and run HiSilicon's nnie models remotely through the internet. The program has been tested
on Hi3516/Hi3519 devices.

# Usage
* Build
```shell script
mkdir build
cmake ..
make
```
* Run the `nnie_transfer_server` on the Hi3516/3519 board.
```shell script
nnie_transfer_server # by default, it listens on port 7777
nnie_transfer_server 8888 # you can specify the port like this
```

# Limitations
* Currently it supports U8 or YUV inputs, S32,VEC or SEQ inputs are not yet supported
* Net types like `ROI/PSROI` and `Recurrent` are not supported.