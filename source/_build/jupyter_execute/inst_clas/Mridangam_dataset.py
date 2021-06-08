#!/usr/bin/env python
# coding: utf-8

# The Mridangam Stroke Dataset
# ============================
# 
# The MTG proposes a few [datasets](https://www.upf.edu/web/mtg/software-datasets) that may be used for the instrument classification task: 
# * [IRMAS](https://www.upf.edu/web/mtg/irmas) - containing annotated samples of multiple instruments 
# * [Good Sounds](https://www.upf.edu/web/mtg/good-sounds) - containing monophonic sounds (notes and scales) for various instruments
# * [Mridangam Stroke](https://compmusic.upf.edu/mridangam-stroke-dataset) containing isolated drum strokes and the annotated tonic
# 
# All these datasets may be downloaded from zenodo. In addition, these datasets may be loaded and validated with easy using the [mirdata](https://github.com/mir-dataset-loaders/mirdata) library. 
# 
# For educational purposes, we will work with the Mridangam Stroke dataset.
# 

# ## Instalation of packages
# 
# To download, validate, and load the data we use the mirdata library.
# We use MTG's [essentia](https://essentia.upf.edu) for audio loading and feature computation. Matplotlib's pyplot is used for plotting and pandas for data stats. 
# We install these libraries through PyPI. 

# In[1]:


get_ipython().run_cell_magic('capture', '', "#If not installed, install Essentia. \n# This cell is for running the notebook in Colab\nimport importlib.util\nif importlib.util.find_spec('essentia') is None:\n    !pip install essentia\n\n!pip install git+https://github.com/mir-dataset-loaders/mirdata.git\n!pip install pandas\n!pip install matplotlib")


# In[2]:


#Basic imports
import os
import matplotlib.pyplot as plt
import numpy as np

# Imports to support MIR
import mirdata
import essentia.standard as ess
import pandas as pd



# ## Dataset description
# ### Data downloading, validation

# In[3]:


#Import Mridangam Stroke Dataset
mridangam_stroke = mirdata.initialize('mridangam_stroke')

#This cell downloads and validates the mridangam dataset
mridangam_stroke.download()  # download the dataset
mridangam_stroke.validate()  # validate that all the expected files are there


# In the mirdata library the track ids in the dataset can be seen retrieved using the *track_ids* attribute. The *load_tracks* methods loads all the tracks in a dictionary. 

# In[4]:


mridangam_ids = mridangam_stroke.track_ids  # Load Mridangam IDs
mridangam_data = mridangam_stroke.load_tracks()  # Load Mridangam data

mridangam_data[mridangam_ids[0]]  # Visualize a single track


# In[5]:


# Get complete list of different strokes
stroke_names = []
for i in mridangam_ids:
    stroke_names.append(mridangam_data[i].stroke_name)
stroke_names = np.unique(stroke_names)

print(stroke_names)


# In[6]:


# You can create a dictionary using stroke type as keys
stroke_dict = {item: [] for item in stroke_names}
for i in mridangam_ids:
    stroke_dict[mridangam_data[i].stroke_name].append(mridangam_data[i].audio_path)

stroke_dict['bheem']


# In[7]:


# Raw-data preprocess analysis parameters
_, fs = mridangam_data[mridangam_ids[0]].audio

num_strokes = len(stroke_dict.keys())
print("Plot waveforms of random samples of each stroke type...")
plt.figure(1, figsize=(5 * num_strokes, 3))
file_ind_inlist = 0 # 0: let's take the first file in the list for sample plots
for i, stroke in enumerate(stroke_dict.keys()):
    sample_file = stroke_dict[stroke][file_ind_inlist]
    x = ess.MonoLoader(filename = sample_file, sampleRate = fs)()
    
    plt.subplot(1,num_strokes,(i+1))
    plt.plot(x)
    plt.title(stroke)

