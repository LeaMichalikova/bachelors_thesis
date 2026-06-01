# Bachelor's thesis
This repository contains the source code, scripts, results and other relevant material developed as a part of our bachelor's thesis "Traffic Camera Calibration Via Detection of Vanishing Points of Vehicles" (in Slovak: "Kalibrácia dopravnej kamery pomocou detekcie úbežníkov vozidiel").

This project is based on the original DeepVPCalib project, developed by Viktor Kocur and Milan Ftáčnik:
https://github.com/kocurvik/deep_vp

Our thesis extends this original project by implementing a series of modifications.

## Previewing Our Results
To preview our results, download the contents of this repository and run the following command from the `deep_vp_bachelors_modifications` directory:
```
python -m eval.eval_calib \  
  ../2016-ITS-BrnoCompSpeed \  
  ../BrnoCarPark
```
Note that despite their name, the `2016-ITS-BrnoCompSpeed` and `BrnoCarPark` directories do not contain the full contents of the datasets.

The results of executed modifications have the following experiment numbers, based on which the relevant files can be identified:

| Modification number | Experiment number|
|---------------|--------------------|
| 1             | 6                  |
| 2             | 7                  |
| 2             | 8                  |
| 3             | 9                  |
| 4             | 10                 |
| 5             | 11                 |

## Running Our Code

All commands used to obtain our results can be found in the `deep_vp_bachelors_modifications/scripts/2026_modifications_commands.sh` file. For our project, we did not run the object detection, since we used the vehicle detections from the original project. They can be found in this repository. Additionally, to run the commands, the BoxCars116k, BrnoCarPark and BrnoCompSpeed datasets are required. As described in our thesis, they are not included in this repository. The BoxCars116k dataset is publicly available at https://github.com/JakubSochor/BoxCars, while the other two datasets can be obtained by contacting their respective authors.