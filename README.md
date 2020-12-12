# High resolution RADAR road segmentation using weakly supervised learning

Autonomous driving has recently gained significant attention due to its disruptive potential and impact on global economy. However, these high expectations are hindered by strict safety requirements for redundant sensing modalities each able to independently perform complex tasks to ensure reliable operation. 
At the core of an autonomous driving algorithmic stack is road segmentation which is the basis for numerous planning and decision-making algorithms. RADAR-based methods fail in many driving scenarios, mainly since various common road delimiters barely reflect RADAR signals, coupled with lack of analytical model of road delimiters and inherit limitations in RADAR angular resolution.
Our approach is based on RADAR data in the form of 2D complex range-Doppler array as input to a Deep Neural Network (DNN) trained to semantically segment the drivable area using weak-supervision from a camera. In addition, guided back propagation was utilized to analyze RADAR data and design a novel perception filter. To our knowledge, our approach allows for the first time the ability to perform road segmentation in common driving scenarios based solely on RADAR data and propose to utilize this method as an enabler for redundant sensing modalities for autonomous driving.



# License

This project is subject under license detailed in license.md and should be used for non-commercial, research purposes only.

