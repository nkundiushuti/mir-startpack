Methods
=============


Building large sound libraries of instrument samples such as the ones used by music makers or producers may require labelling the sounds with the corresponding instrument labels.
Musical instrument recognition or classification {cite}`eronen2000musical` is a Music Information Retrieval (MIR) task that aims at classification of audio signals into instrument-related classes.

This task becomes more complex if the samples contain multiple instruments {cite}`bosch2012comparison` or if we are asked to locate the time frames where an instrument is active within a longer time span {cite}`schluter2015exploring`.


(methods:feature-extractor)=
## Feature extractor approach

Traditionally, MIR systems include a part of feature extraction followed up by a classification part. For instrument classification, timbre-related features, time-frequency representations, such as MFCCs, are extracted from audio signals {cite}`herrera2003automatic`.
The features are then used as input to train an off-the-shelf machine learning classifier which predicts the most likely class for each training sample.


(methods:end-to-end)=
## End-to-end approach

The rise of deep learning has seen also a paradigm shift in MIR. End-to-end approaches using as input the audio signals or time-frequency representations outperformed the feature engineering approaches {cite}`solanki2019music`.  

In terms of machine learning approaches, neural networks architectures designed for audio and music signals replaced the off-the-shelf machine learning methods. These architectures are designed to take as input either the audio signal, spectrograms computed from the audio, the MFCCs or representations learned from the audio.
