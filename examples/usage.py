# pip3 install git+https://github.com/thewh1teagle/nakdimon
# wget https://github.com/elazarg/nakdimon/raw/master/models/Nakdimon.h5

import nakdimon 
import nakdimon.predict


result = nakdimon.predict("Nakdimon.h5", "שלום עולם!")
print(result)