# Local-API-Host

Basically what it says on the tin. Developing this as a dependency for future AI projects so I don't have to pay for openai tokens. That being said. It'll likely be slow but that's fine.

## CUDA

Unless you want agonizingly slow CPU execution, you'll need to get CUDA on your workstation. Start at this wikipedia page and determine if your GPU's compute capability: https://en.wikipedia.org/wiki/CUDA#GPUs_supported. 

From there you'll want to check what cuda version your GPU can handle based on its compute capability here: https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions. 

Finally, if your GPU is supported, you can install cuda here: https://developer.nvidia.com/cuda-downloads. 

## Python Setup
I've developed this using python 3.12. So no guarentees it'll work on lower versions. I reccomend setting up a virtual environment using `python -m venv venv` (you may need to run this command with python3 if you haven't set up an alias) and then activating it using `venv/bin/activate.ps1` on windows or `source venv/bin/activate` on linux. If you're using a different version of python, you'll need to change the version in the requirements.txt file.

## Pip

Now comes the easy part. Install the requirements.txt file using, 
`pip install -r requirements.txt`
To ensure that you get the proper torch install follow the instructions here: https://pytorch.org/get-started/locally/. May you not need to install from source.

That should be all you need to do! If you have any questions, feel free to reach fill out an issue!