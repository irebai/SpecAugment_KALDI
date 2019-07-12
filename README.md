# SpecAugment with KALDI
## A C++ Implementation of SpecAugment paper within KALDI toolkit: A Simple Data Augmentation Method for Automatic Speech Recognition

[SpecAugment](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) is a state of the art data augmentation approach for speech recognition.


SpecAugment idea aims to construct an augmentation policy that acts on the log mel spectrogram directly.
Three transformations are proposed by the paper's authors, which are: time warping, frequency perturbation, and time perturbation.
* "Motivated by the goal that these features should be robust to deformations in the time direction, partial loss of frequency information and partial loss of small segments of speech"

We implemented the frequency and time masking transforms using Kaldi, which is a state-of-the-art automatic speech recognition (ASR) toolkit, containing almost any algorithm currently used in ASR systems.

## To use:
1. Replace the files "feature-common.h, feature-common-inl.h, feature-mfcc.cc, feature-mfcc.h" in the kaldi/src/feat directory by those in our repository

2. Copy the folder src/featbin-mask in the kaldi/src directory

3. Add featbin-mask into the list of "SUBDIRS" in kaldi/src/Makefile file

Example:
```
SUBDIRS = featbin feat featbin-mask
```

4. Run make to re-compile the modified functions

After the install step runs, you should add the kaldi/src/featbin-mask directory to the PATH variable.

5. Check out compute-mfcc-feats-masks function for the added parameters.

### Augmentations parameters
Time Mask
* --time-mask: The time mask parameter T
* --nbr-time-mask: The number time of mask applied

Frequency Mask
* --frequency-mask: The frequency mask parameter F
* --nbr-frequency-mask: The number of frequency mask applied

### Note
To activate or desactivate a parameter, you need only to put the correponding mask to zero:
eg., --time-mask=0; --frequency-mask=0


### Links
- linkedin: https://www.linkedin.com/in/ilyes-rebai-726a392a/
- email: ilyes.rebai@gmail.com
