
import cpuinfo
from dotenv import load_dotenv

load_dotenv()
try:
    processor_type = cpuinfo.get_cpu_info()['brand_raw']
    if "intel" in processor_type.lower():
        from sklearnex import patch_sklearn
        print(f"Using intel-sklearn optimization because processor type is {processor_type}")
        patch_sklearn()
    else:
        print(f"Cannot use intel-sklearn optimization because processor type is {processor_type}")
except Exception as e:
    # print(f"Something went wrong => {e}")
    # # https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html
    # print("Try to install intel-AI-Toolkit from 'https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt'")
    # print("Install command is: sudo apt install intel-basekit")
    # print("Then choose the installed python package with command: 'poetry use env <PATH-TO-INTEL-PYTHON-BINARY>'")
    pass
            
from .common import *
from .confidence_intervall_estimator import *
from .dataset_loader import *
from .saver import *
