## checkpoint download
Depth-Anything-V2-Small https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true

Depth-Anything-V2-Base https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true

Depth-Anything-V2-Large https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

下载后放入checkpoints文件夹里面

## use
先运行depth_server.py
```
python depth_server.py
```
然后运行depth_clint.py
```
python depth_clint.py
```
图片是放到clint里面的路径