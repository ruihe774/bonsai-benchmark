import os,sys
os.environ['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
os.environ['OPENBLAS_NUM_THREADS']='1'
exec(compile(open('tools/check_abi.py').read(),'tools/check_abi.py','exec'))
