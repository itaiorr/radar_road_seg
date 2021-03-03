# High resolution radar road segmentation using weakly supervised learning

[![DOI](https://zenodo.org/badge/320796887.svg)](https://zenodo.org/badge/latestdoi/320796887)

## Abstract
Autonomous driving has recently gained significant attention due to its disruptive potential and impact on global economy. However, these high expectations are hindered by strict safety requirements for redundant sensing modalities each able to independently perform complex tasks to ensure reliable operation. 
At the core of an autonomous driving algorithmic stack is road segmentation which is the basis for numerous planning and decision-making algorithms. Radar-based methods fail in many driving scenarios, mainly since various common road delimiters barely reflect radar signals, coupled with lack of analytical model of road delimiters and inherit limitations in radar angular resolution.
Our approach is based on radar data in the form of 2D complex range-Doppler array as input to a Deep Neural Network (DNN) trained to semantically segment the drivable area using weak-supervision from a camera. In addition, guided back propagation was utilized to analyze RADAR data and design a novel perception filter. To our knowledge, our approach allows for the first time the ability to perform road segmentation in common driving scenarios based solely on radar data and propose to utilize this method as an enabler for redundant sensing modalities for autonomous driving.

## Publication 
https://www.nature.com/articles/s42256-020-00288-6

## Publicly available version
https://www.nature.com/articles/s42256-020-00288-6.epdf?sharing_token=cbVIpW3D9XH24021jMN9J9RgN0jAjWel9jnR3ZoTv0PFDQt3hgiccGGeWGZ3fl_GDw8luPQiL3MFGFA2GUMu_S1hmJWI_xTcysPJaMOZ661o7QB44pOxwyhTMgGnSbD6SPV-7lYL7rjCU7cDu9MR81b1w7JFUhmLjQ-yRjFTrJM%3D

## Dependencies
Detailed in imports.py 

## Citation
For usage of the package and associated manuscript, please cite: 
Orr, I., Cohen, M. & Zalevsky, Z. High-resolution radar road segmentation using weakly supervised learning. Nat Mach Intell (2021). https://doi.org/10.1038/s42256-020-00288-6

## License
This project is covered under license detailed in license.md and should be used for non-commercial, research purposes only.

