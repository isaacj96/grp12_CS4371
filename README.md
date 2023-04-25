# Group 12 CS 4371 Project

## Summary

This project is an extension of the research paper "Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security" [1]. It uses deep learning to train a model of an intrusion detection system against various types of threats.

## Dependencies

This code relies on pandas, numpy, sklearn, tensorflow, and keras

## How to Run

The code for the project is contained within the 'G12.ipynb' notebook and 'classical.py' script. Since the notebook uses deep learning, which is very computationally heavy, it requires the use of a TPU, which we ran using Google Colab. The classical.py script on the other hand can be run using any python interpreter (It will however take a very long time). 

To do so, first download the classical.py script and and the data used to train the models (KDDTest+.csv and KDDTrain+.csv). Then, edit lines 34 and 35 of the script to point to the path of the datasets.

EX: 
```
trainset = pd.read_csv('KDDTrain+.csv', header=0)
testset = pd.read_csv('KDDTest+.csv', header=0)
```

The script can then be ran with the output appearing in the console.

## Data

The data that was used to train the deep learning model can be found [insert repo here]. The data used for the classical ML algorithms is an updated version of the KDDCup99 dataset sourced from the [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)

## Sources

[1] Rahul, V.K., Vinayakumar, R., Soman, K.P., & Poornachandran, P. (2018). Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security. 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT), 1-6.

[2] Rahul-Vigneswaran, K., Poornachandran, P., & Soman, K.P. (2019). A Compendium on Network and Host based Intrusion Detection Systems. CoRR, abs/1904.03491.

