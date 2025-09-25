.. scLinguist documentation master file, created by
   sphinx-quickstart on Mon Jun 23 17:48:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scLinguist documentation
===============================

``scLinguist`` a transformer-based framework for RNA-to-protein prediction that follows a two-step training strategy. First, we pretrain modality-specific models on large-scale single-omics datasets using self-supervised learning to extract informative representations. Subsequently, we fine-tune the model on paired RNA-protein data, enabling accurate cross-modality translation. This training paradigm allows our model to leverage abundant single-omics data while effectively learning modality relationships from limited paired datasets. 

.. figure:: /workflow.png
   :width: 80%
   :align: center

   Overview of scLinguist workflow

Getting started with scLinguist
----------------------------------

To begin using **scLinguist**, please refer to the following sections of the documentation:

- The `Installation <./install.rst>`_ provides instructions for setting up **scLinguist** in your environment.

- The `Tutorials <./tutorials/index.rst>`_ contains examples on how to use **scLinguist**.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   install
   tutorials/index
   datasets
   credits

