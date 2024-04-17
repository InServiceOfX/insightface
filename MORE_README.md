## Creating and starting a virtual environment for Python 3

Create a directory for a virtual environment:

```
/InServiceOfX$ python3 -m venv ./venv/
```

Activate it:
```
/InServiceOfX$ source ./venv/bin/activate
```
You should see the prompt have a prefix `(venv)`.

Deactivate it:
```
deactivate
```

## ONNX runtime but with CUDA 12.x

Needed by `python-package/insightface/__init__.py`

```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```