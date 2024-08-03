# pip install git+https://github.com/nakdimon/nakdimon.git
# mkdir models
# wget https://github.com/elazarg/nakdimon/raw/master/models/Nakdimon.h5
# mv Nakdimon.h5 models/Nakdimon.h5

from nakdimon import diacritize

result = diacritize("שלום עולם!", "models/Nakdimon.h5")
print(result)
