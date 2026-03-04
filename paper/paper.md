---
title: 'VST: A Python-based deep learning tool for segmenting electron microscopy samples'
tags:
  - Python
  - Semantic segmentation
  - Instance segmentation
  - Machine learning
  - Image analysis
  - Microscopy image
  - Three-dimensional image
authors:
  - name: Yuyao (Daniel) Huang
    orcid: 0009-0007-7171-5029
    affiliation: 1
  - name: Nickhil Jadav
    orcid: 0009-0009-2754-1097
    affiliation: 1
  - name: Georgia Rutter
    orcid: 0009-0005-6609-0576
    affiliation: 1
  - name: Mihnea Bostina
    orcid: 0000-0003-3621-3772
    affiliation: "1, 2"
  - name: Duane Harland
    orcid: 0000-0002-1204-054X
    affiliation: 3
  - name: Lech Szymanski
    orcid: 0000-0002-5192-0304
    affiliation: 1
affiliations:
  - name: University of Otago, New Zealand
    index: 1
    ror: 01jmxt844
  - name: Okinawa Institute of Science and Technology Graduate University, Japan
    index: 2
    ror: 02qg15b79
  - name: AgResearch Group, New Zealand Institute for Bioeconomy Science Ltd, New Zealand
    index: 3
    ror: 03j13xx78
bibliography: paper.bib
date: 2 December 2025
---

# Summary
Volume Segmentation Tool (VST) is a Python based deep learning tool designed specifically to segment three-dimensional VEM biological data without extensive requirements for cross disciplinary knowledge in deep learning. The tool is made accessible through a user-friendly interface with visualisations and a one-click installer.

Recognising the current rapid expansion of the VEM field, we have built VST with flexibility and instance segmentation in mind, hoping to ease and accelerate statistical analysis of large datasets in biological and medical research contexts. VST is composed of two main parts: the PyTorch [@paszke2019pytorch]-based deep learning core that performs semantic/instance segmentation on volumetric grey scale image datasets, and a user interface that operates on top of it, responsible for constructing CLI commands to the core components for tasking. The general pipeline of VST is shown in Figure 1. We had put in efforts to ensure VST could automatically handle issues associated with large dataset sizes, instance segmentation, anisotropic voxels and imbalanced classes.

![Schematic diagram for VST](Figure 1.png)

# Statement of need
Volume Electron Microscopy (VEM) enables the capture of 3D structure beyond planar samples, which is crucial for understanding biological mechanisms. With automation, improved resolution, and increased data storage capacity, VEM has led to an explosion of large three-dimensional datasets. Large datasets offer the opportunity to generate statistical data, but analysing them often requires assigning each voxel (3D pixel) to its corresponding structure, a process known as image segmentation. Manually segmenting hundreds or thousands of image slices is tedious and time-consuming. Computer-aided, especially Machine Learning (ML) based segmentation is now a routinely used method, with Trainable Weka Segmentation [@arganda2017trainable] and Ilastik [@berg2019ilastik] being two leading options. Emerging methods for EM image segmentation are often based on Deep Learning (DL) [@mekuvc2020automatic] because this approach has potential to outperform traditional ML in terms of accuracy and adaptivity [@minaee2021image; @erickson2019deep].

Many earlier DL tools developed are highly specific to single sample types, like in connectomics [@li2017compactness; @kamnitsas2017efficient], MRI [@milletari2016v] or X-ray tomography [@li2022auto], they use a subject-optimised design at the cost of adaptability to non-target datasets. Dedicated DL segmentation tools for generalised VEM data are gradually becoming available but each have short-comings. One example, CDeep3M [@haberl2018cdeep3m], which uses cloud computing. Although easy to use, it was designed for anisotropic data (where the z-resolution is much lower than xy-resolution) which creates limitations when applied to isotropic data [@gallusser2022deep]. Another example is DeepImageJ [@gomez2021deepimagej], which runs on local hardware and integrates easily with the ImageJ suit [@schneider2012nih]. However, it only supports pre-trained models and does not have the functionality to train new ones. ZeroCostDL4Mic [@von2021democratising], utilises premade notebooks running on Google Colab, but it requires user interaction during the entire segmentation process, which can take hours and thus is inconvenient. A more recent and advanced example is nnU-Net [@isensee2021nnu], which auto-configurates itself based on dataset properties and has a good support for volumetric dataset, but it focuses exclusively on semantic segmentation and lacks a user friendly interface.

In short, there is a lack of tools that can handle a wide range of VEM data well for generating both semantic and instance segmentation, while at the same time been easy to use, scalable and can be run locally. Which is what motivated us to develop VST - an easy-to-use and adaptive DL tools specifically optimised for generalised VEM image segmentation.

When VST was developed, BiaPy [@franco2025biapy] and DL4MicEverywhere [@hidalgo2024dl4miceverywhere] emerged as competitive options that satisfied the above criteria. They share features such as a GUI and semantic and instance segmentation on 3D images. Apart from not relying on Docker and specialising solely in 3D semantic and instance segmentation, VST incorporates more recent and sophisticated deep learning techniques to improve speed and accuracy (see the sections below).

# Software design
The core principles of VST lie in the user-friendliness and scalability. The software comes with a one-click installer and full documentation on all user-accessible features, with the aim to enable accessibility for domain experts without machine learning expertise. In terms of scalability, VST uses Zarr[@abernathey2026zarr], a framework for distributed storage, which allows just-in-time, chunked access for datasets much larger than the user's system memory. Which is a common situation within VEM, where datasets of hundreds or thousands of gigabytes scales are present.

In terms of design, the software provides a graphical interface over a set of scripts handling various aspects of the deep learning and inference workload. The internal training framework utilises PyTorch [@paszke2019pytorch], the interface compiles terminal commands and activates the Python scripts as needed (Figure 1). For the DL model underneath, VST uses a 3D version of Swin transformer [@liu2021swin] for the DL model underneath. This is a proven DL architecture for a wide range of vision tasks and is known for its superior performance to the traditional U-Net [@ronneberger2015u] architecture, as well as its linear computational complexity. The size, depth, and other details of the model are configured automatically based on the characteristics of the user's dataset. Another recent technique that VST has integrated is the use of adaptive muon [@si2025adamuon] as the optimiser due to its faster convergence.

Much of VST's internal logic is optimised for single-class semantic and instance segmentation and cannot easily be transferred to the multi-class case or 2D image segmentation. This trade-off was made to keep workloads manageable, simplify the codebase, and maximise the ease of use for those with minimal machine learning expertise.

# Research impact statement
VST has been used in postgraduate projects at the University of Otago in New Zealand for segmentation of the entire mitochondrial complement of tumorsphere [@jadav2023beyond], as well as poorly demarked cell remnants within wool fibres. VST's competitive performance to nnU-Net [@huang2025generalist], an MIT open-source licence, and comprehensive documentation make it ready for use by the wider community.

# The graphical user interface
VST's GUI is supported by the Gradio package [@abid2019gradio] and hosted on the user's browser.

The GUI is divided into three sections: Main, Activations Visualisation and Extras.

The main section (Figure 2) contains settings regarding training and using segmenting networks. Two segmentation modes are supported: semantic segmentation, in which the foreground objects are separated from the background, and instance segmentation, in which individual foreground objects are separated from each other as well. User can either train a new network, load an existing network and use it for predictions on new data, or train one and use it immediately.

Upon training, it automatically opens a TensorBoard interface [@pang2020deep] to provides various real time visualisations for the training process.

![The main interface of VST](Figure 2.png)

The activations visualisation section requires a trained network and an example image. Given that image, it plots the activation across each channel through all layers of the network.

The extra section contains two functionalities: exporting the TensorBoard log to an Excel table, calculating segmentation metrics between (potentially) generated labels and ground truth labels.

# AI Usage Disclosure
ChatGPT and DeepSeek were used to generate some Python functions. All generated functions were thoroughly analysed, tested with real-world data, modified and verified to satisfy desired input and output conditions. No generative AI tools were used in the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements
We want to thank to the New Zealand Institute for Bioeconomy Science for their support during the MSc study and University of Otago Postgraduate Publishing Bursary (Master's) for their support with the time of writing.

# References