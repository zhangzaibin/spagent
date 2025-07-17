# Intro
- depth_client.py 是真实的client，需要部署完之后从depth_client中导入infer函数，在workflow里面进行调用。
- mock_depth_service.py 一个假的depth_client，用来调试用的，后面使用真的这个会被舍弃掉。


# demploy server on another machine
## checkpoint download
Depth-Anything-V2-Small https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

Depth-Anything-V2-Base https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

Depth-Anything-V2-Large https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

下载后放入checkpoints文件夹里面

## use
先运行depth_server.py
```
python depth_server.py
```
然后运行depth_client.py
```
python depth_client.py
```
图片是放到client里面的路径
