# Static Eccentricity Fault Location Diagnosis in Resolvers Using Siamese-Based Few-Shot Learning
the code repository for the paper "Static Eccentricity Fault Location Diagnosis in Resolvers Using Siamese-Based Few-Shot Learning"

[Paper Link in IEEE Xplore](https://ieeexplore.ieee.org/document/10192453)

## citation
```
@ARTICLE{10192453,
  author={Emadaleslami, Mahdi; KhajueeZadeh, MohammadSadegh; and Tootoonchian, Farid},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Static Eccentricity Fault Location Diagnosis in Resolvers Using Siamese-Based Few-Shot Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIM.2023.3298404}

```

## Env requirement
Python 3.6 and the following versioned packages were used for All codes:
- Keras >= 2.8.0
- matplotlib >= 3.3.4
- seaborn >= 0.11.1
  
open main _Code.ipynb by **Jupyter Notebook**  for better experience.
## Structure

- **Datasets**: The 2D TS-FEA is used to create Data set by injecting static eccentricity fault in the resolver, where the excitation voltage, excitation frequency, and angular velocity are 5V, 5kHz, and 480 rpm. Raw voltage signals based on slot location are in the "Raw Signal Voltage" folder
- **Complete_raw_signal_Data.xlsx**:  all raw voltage signals as one file for further use with the pandas library
- **main _Code.ipynb**: the main file for usage
- **models.py**: Define the Siamese Network model and other functions.
- **siamese_1D.py**: Define the init of Siamese Network input data, training, and testing functions.
- **Pictures**:  code for Confusion Matrix & T-SNE

## Intrudoction
While Permanent Magnet Synchronous Motors (PMSMs) are widely used in Electric Vehicles (EVs) due to high efficiency and reliability, the successful drive of PMSMs highly depends on the rotor angle position. Among various methods to detect rotor angle, Resolver is favored because of high durability and fault tolerance against dust and vibrations. However, it is sensitive to electromagnetic, electric, and aerodynamic fault, especially static eccentricity; therefore, This paper focuses on static eccentricity fault location diagnosis with limited data.

Recently, few-shot learning has been suggested to electromagnetic systems fault as an exciting branch of AI to overcome the data scarcity challenges. Despite the fact that few-shot learning has high accuracy, it has yet to be regarded as a resolver diagnosis. Accordingly, this study proposes a few-shot learning Convolutional Neural Network (CNN) to diagnose the static eccentricity location with raw signal voltages. Given the raw signal voltage has minor changes between healthy and faulty conditions, analyzing its signatures is extremely hard, as shown in Fig. 1. The suggested few-shot learning model is based on the siamese neural network, which learns by exploiting sample pairs of the same or different categories. Our model consists of three layers: data, training, and application, as shown in Fig. 2. Accordingly, the data layer collects raw sin/cosine signal voltages of injected static eccentricity under each stator tooth. Then, the training layer directly takes raw signal pairs with location tags, either the same or different, to extract features. Subsequently, a Siamese network learns through an iterative manner to yield the probability of signal pairs being the same or not. Consequently, in the application layer, the Siamese network takes raw signal voltages of query and support sets to diagnose the static eccentricity location based on the support set signal that yields the greatest similarity. 


![.](https://github.com/mahdiemad/Static-Eccentricity-Fault-Location-Diagnosis-in-Resolvers-with-Few-Shot-Learning/assets/57590076/6191aa16-86d8-4708-b453-31b4da45d1fc)
#### Fig. 1. Signal voltages of sine winding when static eccentricity location is under each of twelve teeth (winding groups are shown in four colors, including blue, orange, red, and green)

![Fig.2.](https://github.com/mahdiemad/Static-Eccentricity-Fault-Location-Diagnosis-in-Resolvers-with-Few-Shot-Learning/assets/57590076/a7c28eb8-36d0-461b-9f87-a5030474c491)
#### Fig. 2. Overview of the suggested few-shot learning Siamese network for the static eccentricity location diagnosis      

Also, The pseudo-code is as follows:

![psedo](https://github.com/mahdiemad/Static-Eccentricity-Fault-Location-Diagnosis-in-Resolvers-with-Few-Shot-Learning/assets/57590076/f5c4c220-1236-4935-a6bc-58d974a62c4c)
 

All our models in this study are open source. But, due to privacy concerns, only part of our dataset is publicly available, which should be enough for educational purposes.

## acknowledgement
Finally, we would like to thank  authors of the paper "Limited Data Rolling Bearing Fault Diagnosis with Few-shot Learning". Enthusiastic readers are encouraged to review  [the paper](https://ieeexplore.ieee.org/abstract/document/8793060) at IEEE Explore.

