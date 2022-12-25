
import platform
import cpuinfo

try:
    processor_type = cpuinfo.get_cpu_info()['brand_raw']
    if "intel" in processor_type.lower():
        from sklearnex import patch_sklearn
        print(f"Using intel-sklearn optimization because processor type is {processor_type}")
        patch_sklearn()
    else:
        print(f"Cannot use intel-sklearn optimization because processor type is {processor_type}")
except Exception as e:
    print(f"Something went wrong => {e}")
    # https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html
    if "libsvml.so" in str(e):
        print("Try to install 'icc-rt'")
    if "libsycl.so" in str(e):
        print("Try to install 'opencl-rt'")
            

import sklearn
from .common import *
from .dataset_loader import *
from .saver import *
from .confidence_intervall_estimator import *
