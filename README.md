# Nakdimon: a simple Hebrew diacritizer

Repository for the paper [Restoring Hebrew Diacritics Without a Dictionary](https://arxiv.org/abs/2105.05209) by Elazar Gershuni and Yuval Pinter.

Demo: https://nakdimon.org/

## Building and running docker container
Build the docker container:
```
$ docker build -t nakdimon .
```

Run the docker container:
```
$ docker run --rm --gpus all --user 1000:1000 -it nakdimon /bin/bash
```

The `--gpus all` flag is required to run the container with GPU support.

## Training and evaluating
To train, test and evaluate the system, run the following commands:
```
> python nakdimon train --model=models/Nakdimon.h5
> python nakdimon run_test --test_set=tests/new --model=models/Nakdimon.h5
> python nakdimon results --test_set=tests/new --systems Snopi Morfix Dicta MajAllWithDicta Nakdimon
```
The first step trains the model and create a file named `Nakdimon.h5` in the `models` directory.
By default, the model is the one described in the paper: `models/Nakdimon.h5`.
If the model already exists, you may skip this step. 

The second step asks the Nakdimon server to predict the diacritics for the test set. You may skip this step.
A folder for the results is created in the chosen test folder, with the same name as the model; in this case, `tests/new/NakdimonNew`.
By default, the test set is the one used in the paper (`tests/new`); you can use `tests/dicta` instead.
If the test results already exist, you may skip this step. If you are not sure, you can use the `--skip_existing` flag.

The third step calculates and prints the results (DEC, CHA, WOR and VOC metrics, as well as OOV_WOR and OOV_VOC).
By default, the systems are the folders in the chosen test folder.
For the Dicta test set (`/tests/dicta`) you should use `MajAllNoDicta` instead of `MajAllWithDicta`, otherwise the vocabulary for the Majority would include the test set itself.

## Diacritizing a single file
```
> python nakdimon predict input_file.txt output_file.txt
```

## Using other systems
You can use the `run_test` command to run the test set on other systems, such as Dicta:
```
> python nakdimon run_test --test_set=tests/new --system=Dicta
```
This will create a folder named `Dicta` for the results in the `tests/new` folder.
Note that `Morfix` cannot be used in this manner, as its license prohibit automatic use.

## Running ablation tests
You can use the `--ablation` flag to train different models for the ablation tests and other experiments:
```
> python nakdimon train --model=models/SingleLayer.h5 --ablation=SingleLayer
```
See the file `ablation.py` for the list of available ablation parameters.

## Important folders
* `hebrew_diacritized` is the training set.
* `tests` contains three tests sets: `new`, `dicta` and `validation`.
  Each test set has an `expected` folder that describes the ground truth.
  The results of `python nakdimon run_test` are stored in sibling folder, named after the model.
* `models` contains the trained model.
* `nakdimon` holds the source code.

## Citation
```
@inproceedings{gershuni2022restoring,
  title={Restoring Hebrew Diacritics Without a Dictionary},
  author={Gershuni, Elazar and Pinter, Yuval},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  pages={1010--1018},
  year={2022}
}
```
> Gershuni, Elazar, and Yuval Pinter. "Restoring Hebrew Diacritics Without a Dictionary." Findings of the Association for Computational Linguistics: NAACL 2022. 2022.
