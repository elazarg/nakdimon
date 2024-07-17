# pip install git+https://github.com/thewh1teagle/nakdimon@feat/python-package
# wget https://github.com/elazarg/nakdimon/raw/master/models/Nakdimon.h5

import nakdimon

result = nakdimon.predict("Nakdimon.h5", "שלום עולם!")
print(result)