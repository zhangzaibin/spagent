# Intro
- depth_server.py 是server端，运行client或者workflow之前需要先运行这个。
- depth_client.py 是真实的client，需要部署完之后从depth_client中导入infer函数，在workflow里面进行调用。
- mock_depth_service.py 一个假的depth_client，用来调试用的，后面使用真的这个会被舍弃掉。
- _service.py 集成了server和clinet的类，可以直接运行进行测试。暂时不知道后续会不会用上。


# demploy server on another machine
## checkpoint download
Depth-Anything-V2-Small [link](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth)

Depth-Anything-V2-Base [link](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth)

Depth-Anything-V2-Large [link](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)

下载后放入checkpoints文件夹里面

## check server
检查server是否能正常运行。
1. 先运行depth_server.py
```
python depth_server.py
```
2. 运行depth_client.py
```
python depth_client.py
```
图片是放到client里面的路径
