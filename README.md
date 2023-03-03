
<h1 align="center">Awesome open data-centric AI</h1>
<p align="center">Open source tooling for data-centric AI on unstructured data</p>

<p align="center">
	<img src="https://camo.githubusercontent.com/64f8905651212a80869afbecbf0a9c52a5d1e70beab750dea40a994fa9a9f3c6/68747470733a2f2f617765736f6d652e72652f62616467652e737667" alt="Awesome" data-canonical-src="https://awesome.re/badge.svg" style="max-width: 100%;">	     
</p>

<p align="center"><a href="https://github.com/Renumics/awesome-open-data-centric-ai"><img src="static/img/dcai_landscape_s.png" width="100%"/></a></p>


**Data-centric AI (DCAI)** is a development paradigm for ML-based solutions. The term was coined by Andrew Ng who gave the following definition:

> Data-centric AI is the practice of systematically engineering the data used to build AI systems. 

At [Renumics](https://renumics.com), we believe DCAI is an important puzzle piece for building real-world AI systems that generate value. We like the following definition:
> Data-centric AI means to improve training datasets systematically and iteratively by leveraging information from trained ML models.

**Tools that can be efficiently used in day-to-day applications** are the most important ingredient for the DCAI paradigm. This curated link collection is intended to help you discover useful open source tools for your data-centric AI workflows.

## 🔎 Scope

We include useful tools that have an **open-source license** and are **actively maintained** in this collection. All tools mentioned are useful for building DCAI workflows on **unstructured data** (e.g. images, audio, video, time-series, text). 

In order to keep a useful focus and to prevent duplicate work, we exclude the following topics:
1. DCAI tools for tabular data. There is an [awesome list](https://github.com/Data-Centric-AI-Community/awesome-data-centric-ai) for that maintained by the [Ydata team](https://github.com/Data-Centric-AI-Community).
2. Labeling tools. Although labeling is part of the DCAI workflow, we refer to the [awesome list](https://github.com/zenml-io/awesome-open-data-annotation) of the [ZenML team](https://github.com/zenml-io) on that topic. 
3. MLOps tooling. There are many gray areas between MLOps and DCAI and some distinctions have yet to be made. We exclude all topics that are clearly out of the DCAI scope (e.g. AutoML, serving, orchestration etc.).



## 📒 Contents


- [Data versioning](#data-versioning)
- [Embeddings and pre-trained models](#embeddings-and-pre-trained-models)
- [Visualization and interaction](#visualization-and-interaction)
- [Outlier and noise detection](#outlier-and-noise-detection)
- [Explainability](#explainability)
- [Active learning](#active-learning)
- [Uncertainty quantification](#uncertainty-quantification)
- [Bias and fairness](#bias-and-fairness)
- [Drift detection and monitoring](#drift-detection-and-monitoring)
- [Augmentation and synthetic data](#augmentation-and-synthetic-data)
- [Adversarial robustness](#adversarial-robustness)

## :open_hands: Contributing
Do you think something is missing? Please help contribute to this list by contacting us or adding a pull request.


# Data versioning

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <img src="https://camo.githubusercontent.com/7a3bb42ff937bf609d1ce4b4626357cd7b9f0208a46c531b7df8d8636d1aa895/68747470733a2f2f7374617469632e6974657261746976652e61692f6c6f676f2f6476632e737667" width="100"/> | [Data version control (DVC)](https://github.com/iterative/dvc) | Data Version Control or DVC is a command line tool and VS Code Extension to help you develop reproducible machine learning projects. | ![GitHub stars](https://img.shields.io/github/stars/iterative/dvc?style=social) | <a href="https://github.com/iterative/dvc/blob/main/LICENSE"><img src="https://img.shields.io/github/license/iterative/dvc" height="15"/></a> |
| <a href="https://github.com/activeloopai/deeplake"><img src="https://camo.githubusercontent.com/d0c805affb06f5ea9ba767de06b77a04de54a7ef433fad08b2729d5e6b11112c/68747470733a2f2f692e706f7374696d672e63632f72736a63576333532f646565706c616b652d6c6f676f2e706e67"  width="100"/></a> | [deeplake](https://github.com/activeloopai/deeplake) | Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets.  | ![GitHub stars](https://img.shields.io/github/stars/activeloopai/deeplake?style=social) | <a href="https://github.com/activeloopai/deeplake/blob/main/LICENSE"><img src="https://img.shields.io/github/license/activeloopai/deeplake" height="15"/></a> |
| <a href="https://github.com/pachyderm/pachyderm"><img src="https://avatars.githubusercontent.com/u/10432478?s=200&v=4"  width="100"/></a> | [Pachyderm](https://github.com/pachyderm/pachyderm) | Pachyderm – Automate data transformations with data versioning and lineage. | ![GitHub stars](https://img.shields.io/github/stars/pachyderm/pachyderm?style=social) | <a href="https://github.com/pachyderm/pachyderm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pachyderm/pachyderm" height="15"/></a> |
| <a href="https://github.com/git-lfs/git-lfs"><img src="https://avatars.githubusercontent.com/u/20246716?s=200&v=4"  width="100"/></a> | [Git Large File Storage](https://github.com/git-lfs/git-lfs) | Git LFS is a command line extension and specification for managing large files with Git.  | ![GitHub stars](https://img.shields.io/github/stars/git-lfs/git-lfs?style=social) | <a href="https://github.com/git-lfs/git-lfs/blob/main/LICENSE"><img src="https://img.shields.io/github/license/git-lfs/git-lfs" height="15"/></a> |
| <a href="https://github.com/treeverse/lakeFS"><img src="https://raw.githubusercontent.com/treeverse/lakeFS/master/docs/assets/img/logo_large.png"  width="100"/></a> | [lakeFS](https://github.com/treeverse/lakeFS) | lakeFS is an open-source tool that transforms your object storage into a Git-like repository. | ![GitHub stars](https://img.shields.io/github/stars/treeverse/lakeFS?style=social) | <a href="https://github.com/treeverse/lakeFS/blob/main/LICENSE"><img src="https://img.shields.io/github/license/treeverse/lakeFS" height="15"/></a> |

# Embeddings and pre-trained models

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/towhee-io/towhee"><img src="https://avatars.githubusercontent.com/u/87362374?s=200&v=4"  width="100"/></a> | [towhee](https://github.com/towhee-io/towhee) | Towhee is a framework that is dedicated to making neural data processing pipelines simple and fast.   | ![GitHub stars](https://img.shields.io/github/stars/towhee-io/towhee?style=social) | <a href="https://github.com/towhee-io/towhee/blob/main/LICENSE"><img src="https://img.shields.io/github/license/towhee-io/towhee" height="15"/></a> |
| <a href="https://github.com/tensorflow/hub"><img src="https://www.tensorflow.org/static/site-assets/images/project-logos/tensorflow-hub-logo-social.png"  width="100"/></a> | [Tensorflow Hub](https://github.com/tensorflow/hub) | TensorFlow Hub is a repository of reusable assets for machine learning with TensorFlow.   | ![GitHub stars](https://img.shields.io/github/stars/tensorflow/hub?style=social) | <a href="https://github.com/tensorflow/hub/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tensorflow/hub" height="15"/></a> |
| <a href="https://github.com/huggingface/transformers"><img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4"  width="100"/></a> | [Huggingface transformers](https://github.com/huggingface/transformers) | State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.   | ![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers?style=social) | <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img src="https://img.shields.io/github/license/huggingface/transformers" height="15"/></a> |
| <a href="https://github.com/lightly-ai/lightly"><img src="https://avatars.githubusercontent.com/u/50146475?s=200&v=4"  width="100"/></a> | [Lightly](https://github.com/lightly-ai/lightly) | Lightly is a computer vision framework for self-supervised learning.    | ![GitHub stars](https://img.shields.io/github/stars/lightly-ai/lightly?style=social) | <a href="https://github.com/lightly-ai/lightly/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lightly-ai/lightly" height="15"/></a> |

# Visualization and Interaction

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/renumics/spotlight"><img src="https://raw.githubusercontent.com/Renumics/spotlight/main/static/img/spotlight.svg"  width="100"/></a> | [Renumics Spotlight](https://github.com/renumics/spotlight) | Curation tool for unstructured data that connects your stack to the data-centric AI ecosystem.    | ![GitHub stars](https://img.shields.io/github/stars/renumics/spotlight?style=social) | <a href="https://github.com/renumics/spotlight/blob/main/LICENSE"><img src="https://img.shields.io/github/license/renumics/spotlight" height="15"/></a> |
| <a href="https://github.com/voxel51/fiftyone"><img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png"  width="100"/></a> | [FiftyOne ](https://github.com/voxel51/fiftyone) | The open-source tool for building high-quality datasets and computer vision models.   | ![GitHub stars](https://img.shields.io/github/stars/voxel51/fiftyone?style=social) | <a href="https://github.com/voxel51/fiftyone/blob/main/LICENSE"><img src="https://img.shields.io/github/license/voxel51/fiftyone" height="15"/></a> |
| <a href="https://github.com/code-kern-ai/refinery"><img src="https://www.kern.ai/refinery-bicon.png"  width="100"/></a> | [refinery](https://github.com/code-kern-ai/refinery) | The data scientist's open-source choice to scale, assess and maintain natural language data.   | ![GitHub stars](https://img.shields.io/github/stars/code-kern-ai/refinery?style=social) | <a href="https://github.com/code-kern-ai/refinery/blob/main/LICENSE"><img src="https://img.shields.io/github/license/code-kern-ai/refinery" height="15"/></a> |
| <a href="https://github.com/argilla-io/argilla"><img src="https://raw.githubusercontent.com/dvsrepo/imgs/main/rg.svg"  width="100"/></a> | [Argilla](https://github.com/argilla-io/argilla) | Argilla helps domain experts and data teams to build better NLP datasets in less time.    | ![GitHub stars](https://img.shields.io/github/stars/argilla-io/argilla?style=social) | <a href="https://github.com/argilla-io/argilla/blob/main/LICENSE"><img src="https://img.shields.io/github/license/argilla-io/argilla" height="15"/></a> |
| <a href="https://github.com/xtreme1-io/xtreme1"><img src="https://avatars.githubusercontent.com/u/120552187?s=280&v=4"  width="100"/></a> | [Xtreme1](https://github.com/xtreme1-io/xtreme1) | Xtreme1 is the world's first open-source platform for multisensory training data.     | ![GitHub stars](https://img.shields.io/github/stars/argilla-io/argilla?style=social) | <a href="https://github.com/xtreme1-io/xtreme1/blob/main/LICENSE"><img src="https://img.shields.io/github/license/xtreme1-io/xtreme1" height="15"/></a> |
| <a href="https://github.com/richardpaulhudson/holmes-extractor"><img src="https://explosion.ai/static/5d374c336e32dab0b18edb0bdc154aa6/09ca8/introduction-to-holmes.webp"  width="100"/></a> | [Holmes Extractor](https://github.com/richardpaulhudson/holmes-extractor) | Holmes supports a number of use cases involving information extraction from English and German texts. In all use cases (chatbot, structural extraction, topic matching, and supervised classification), the information extraction is based on analysing the semantic relationships expressed by the component parts of each sentence. | ![GitHub stars](https://img.shields.io/github/stars/richardpaulhudson/holmes-extractor?style=social) | <a href="https://github.com/richardpaulhudson/holmes-extractor/blob/master/LICENSE"><img src="https://img.shields.io/github/license/lightly-ai/lightly" height="15"/></a> |


# Outlier and noise detection

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/cleanlab/cleanlab"><img src="https://avatars.githubusercontent.com/u/90712480?s=280&v=4"  width="100"/></a> | [Cleanlab](https://github.com/cleanlab/cleanlab) | Cleanlab facilitates machine learning with messy, real-world data by providing clean labels for robust training and flagging errors in your data.  | ![GitHub stars](https://img.shields.io/github/stars/cleanlab/cleanlab?style=social) | <a href="https://github.com/cleanlab/cleanlab/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cleanlab/cleanlab" height="15"/></a> |
| **PyOD** | [PyOD](https://github.com/yzhao062/pyod) | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection)     | ![GitHub stars](https://img.shields.io/github/stars/yzhao062/pyod?style=social) | <a href="https://github.com/yzhao062/pyod/blob/main/LICENSE"><img src="https://img.shields.io/github/license/yzhao062/pyod" height="15"/></a> |
| <a href="https://github.com/datamllab/tods"><img src="https://raw.githubusercontent.com/datamllab/tods/master/docs/img/tods_logo.png"  width="100"/></a> | [TODS](https://github.com/datamllab/tods) | An full-stack automated time-series outlier detection system.     | ![GitHub stars](https://img.shields.io/github/stars/datamllab/tods?style=social) | <a href="https://github.com/datamllab/tods/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datamllab/tods" height="15"/></a> |
| <a href="https://github.com/SeldonIO/alibi-detect"><img src="https://raw.githubusercontent.com/SeldonIO/alibi-detect/master/doc/source/_static/Alibi_Detect_Logo_rgb.png"  width="100"/></a> | [Alibi Detect](https://github.com/SeldonIO/alibi-detect) | Algorithms for outlier, adversarial and drift detection.    | ![GitHub stars](https://img.shields.io/github/stars/SeldonIO/alibi-detect?style=social) | <a href="https://github.com/SeldonIO/alibi-detect/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SeldonIO/alibi-detect" height="15"/></a> |

# Explainability

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/slundberg/shap"><img src="https://shap-lrjball.readthedocs.io/en/latest/_static/shap_logo.png"  width="100"/></a> | [SHAP](https://github.com/slundberg/shap) | A game theoretic approach to explain the output of any machine learning model.     | ![GitHub stars](https://img.shields.io/github/stars/slundberg/shap?style=social) | <a href="https://github.com/slundberg/shap/blob/main/LICENSE"><img src="https://img.shields.io/github/license/slundberg/shap" height="15"/></a> |
| <a href="https://github.com/SeldonIO/alibi"><img src="https://raw.githubusercontent.com/SeldonIO/alibi/master/doc/source/_static/Alibi_Explain_Logo_rgb.png"  width="100"/></a> | [Alibi](https://github.com/SeldonIO/alibi) | Alibi is an open source Python library aimed at machine learning model inspection and interpretation.     | ![GitHub stars](https://img.shields.io/github/stars/SeldonIO/alibi?style=social) | <a href="https://github.com/SeldonIO/alibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SeldonIO/alibi" height="15"/></a> |
| **LIME** | [LIME](https://github.com/marcotcr/lime) | Explaining the predictions of any machine learning classifier.     | ![GitHub stars](https://img.shields.io/github/stars/marcotcr/lime?style=social) | <a href="https://github.com/marcotcr/lime/blob/main/LICENSE"><img src="https://img.shields.io/github/license/marcotcr/lime" height="15"/></a> |
| <a href="https://github.com/pytorch/captum"><img src="https://raw.githubusercontent.com/pytorch/captum/master/website/static/img/captum_logo.png"  width="100"/></a> | [Captum](https://github.com/pytorch/captum) | Model interpretability and understanding for PyTorch.     | ![GitHub stars](https://img.shields.io/github/stars/pytorch/captum?style=social) | <a href="https://github.com/pytorch/captum/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pytorch/captum" height="15"/></a> |

# Active learning

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/modAL-python/modAL"><img src="https://camo.githubusercontent.com/1e424af39166bfa1bb4b49ba00bd39c617f16e4fc0931d78185413fdf720e015/68747470733a2f2f6d6f64616c2d707974686f6e2e72656164746865646f63732e696f2f656e2f6c61746573742f5f7374617469632f6d6f64414c5f622e706e67"  width="100"/></a> | [modAL](https://github.com/modAL-python/modAL) | A modular active learning framework for Python.   | ![GitHub stars](https://img.shields.io/github/stars/modAL-python/modAL?style=social) | <a href="https://github.com/modAL-python/modAL/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modAL-python/modAL" height="15"/></a> |
| <a href="https://github.com/baal-org/baal"><img src="https://camo.githubusercontent.com/a29cf47a24993d08cfdae02dabcaa04602e7fd3cd5bbd8ddf053238373409625/68747470733a2f2f692e696d6775722e636f6d2f5a647a6232515a2e706e67"  width="100"/></a> | [Bayesian Active Learning (Baal)](https://github.com/baal-org/baal) | Library to enable Bayesian active learning in your research or labeling work.     | ![GitHub stars](https://img.shields.io/github/stars/baal-org/baal?style=social) | <a href="https://github.com/baal-org/baal/blob/main/LICENSE"><img src="https://img.shields.io/github/license/baal-org/baal" height="15"/></a> |


# Uncertainty quantification

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/uncertainty-toolbox /"><img src="https://repository-images.githubusercontent.com/293174301/60be5300-463c-11eb-9b69-70a31a38e5ee"  width="100"/></a> | [Uncertainty Toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox/) | A Python toolbox for predictive uncertainty quantification, calibration, metrics, and visualization.    | ![GitHub stars](https://img.shields.io/github/stars/uncertainty-toolbox/uncertainty-toolbox?style=social) | <a href="https://github.com/scikit-learn-contrib/MAPIE/blob/main/LICENSE"><img src="https://img.shields.io/github/license/scikit-learn-contrib/MAPIE" height="15"/></a> |
| <a href="https://github.com/scikit-learn-contrib/MAPIE"><img src="https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/doc/images/mapie_logo_nobg_cut.png"  width="100"/></a> | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | A scikit-learn-compatible module for estimating prediction intervals.     | ![GitHub stars](https://img.shields.io/github/stars/scikit-learn-contrib/MAPIE?style=social) | <a href="https://github.com/scikit-learn-contrib/MAPIE/blob/main/LICENSE"><img src="https://img.shields.io/github/license/scikit-learn-contrib/MAPIE" height="15"/></a> |

# Bias and fairness

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/Trusted-AI/AIF360"><img src="https://artwork.lfaidata.foundation/projects/ai-fairness-360/icon/color/ai-fairness-360-icon-color.png"  width="100"/></a> | [AIF360](https://github.com/Trusted-AI/AIF360) | The AI Fairness 360 toolkit helps to detect and mitigate bias in machine learning models throughout the AI application lifecycle.    | ![GitHub stars](https://img.shields.io/github/stars/Trusted-AI/AIF360?style=social) | <a href="https://github.com/Trusted-AI/AIF360/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Trusted-AI/AIF360" height="15"/></a> |
| <a href="https://github.com/fairlearn/fairlearn"><img src="https://avatars.githubusercontent.com/u/54074260?s=60&v=4"  width="100"/></a> | [Fairlearn](https://github.com/fairlearn/fairlearn) | A Python package to assess and improve fairness of machine learning models.     | ![GitHub stars](https://img.shields.io/github/stars/fairlearn/fairlearn?style=social) | <a href="https://github.com/fairlearn/fairlearn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fairlearn/fairlearn" height="15"/></a> |

# Drift detection

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/deepchecks/deepchecks"><img src="https://avatars.githubusercontent.com/u/92298186?s=200&v=4"  width="100"/></a> | [Deepchecks](https://github.com/deepchecks/deepchecks) | Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.     | ![GitHub stars](https://img.shields.io/github/stars/deepchecks/deepchecks?style=social) | <a href="https://github.com/deepchecks/deepchecks/blob/main/LICENSE"><img src="https://img.shields.io/github/license/deepchecks/deepchecks" height="15"/></a> |
| <a href="https://github.com/evidentlyai/evidently"><img src="https://avatars.githubusercontent.com/u/75031056?s=200&v=4"  width="100"/></a> | [Evidently](https://github.com/evidentlyai/evidently) | An open-source framework to evaluate, test and monitor ML models in production.    | ![GitHub stars](https://img.shields.io/github/stars/evidentlyai/evidently?style=social) | <a href="https://github.com/evidentlyai/evidently/blob/main/LICENSE"><img src="https://img.shields.io/github/license/evidentlyai/evidently" height="15"/></a> |

# Augmentation and synthetic data

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/albumentations-team/albumentations"><img src="https://avatars.githubusercontent.com/u/57894582?s=200&v=4"  width="100"/></a> | [Albumentations](https://github.com/albumentations-team/albumentations) | Fast image augmentation library and an easy-to-use wrapper around other libraries.    | ![GitHub stars](https://img.shields.io/github/stars/albumentations-team/albumentations?style=social) | <a href="https://github.com/albumentations-team/albumentations/blob/main/LICENSE"><img src="https://img.shields.io/github/license/albumentations-team/albumentations" height="15"/></a> |
| <a href="https://github.com/gretelai/gretel-synthetics"><img src="https://camo.githubusercontent.com/d9af85480f0c8d7eef05a377d31b2a4471e83371a92c2fca462943db8dc68bf6/68747470733a2f2f67726574656c2d7075626c69632d776562736974652e73332e616d617a6f6e6177732e636f6d2f6173736574732f676f62735f7468655f6361745f4031782e706e67"  width="100"/></a> | [Gretel Synthetics](https://github.com/gretelai/gretel-synthetics) | Synthetic data generators for structured and unstructured text, featuring differentially private learning.     | ![GitHub stars](https://img.shields.io/github/stars/gretelai/gretel-synthetics?style=social) | <a href="https://github.com/gretelai/gretel-synthetics/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gretelai/gretel-synthetics" height="15"/></a> |
| <a href="https://github.com/sdv-dev/SDV"><img src="https://raw.githubusercontent.com/sdv-dev/SDV/master/docs/images/SDV-DataCebo.png"  width="100"/></a> | [SDV](https://github.com/sdv-dev/SDV) | Synthetic Data Generation for tabular, relational and time series data.     | ![GitHub stars](https://img.shields.io/github/stars/sdv-dev/SDV?style=social) | <a href="https://github.com/sdv-dev/SDV/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sdv-dev/SDV" height="15"/></a> |

# Adversarial Robustness

| Logo | Name | Description | Popularity | License |
| ------- | ---- | ----------- | ---------- | -------- |
| <a href="https://github.com/cleverhans-lab/cleverhans"><img src="https://raw.githubusercontent.com/cleverhans-lab/cleverhans/master/assets/logo.png"  width="100"/></a> | [CleverHans](https://github.com/cleverhans-lab/cleverhans) | An adversarial example library for constructing attacks, building defenses, and benchmarking both.    | ![GitHub stars](https://img.shields.io/github/stars/cleverhans-lab/cleverhans?style=social) | <a href="https://github.com/cleverhans-lab/cleverhans/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cleverhans-lab/cleverhans" height="15"/></a> |
| <a href="https://github.com/Trusted-AI/adversarial-robustness-toolbox"><img src="https://raw.githubusercontent.com/Trusted-AI/adversarial-robustness-toolbox/main/docs/images/art_lfai.png"  width="100"/></a> | [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | Python Library for Machine Learning Security - Evasion, Poisoning, Extraction, Inference - Red and Blue Teams.    | ![GitHub stars](https://img.shields.io/github/stars/Trusted-AI/adversarial-robustness-toolbox?style=social) | <a href="https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Trusted-AI/adversarial-robustness-toolbox" height="15"/></a> |
| <a href="https://github.com/bethgelab/foolbox"><img src="https://raw.githubusercontent.com/bethgelab/foolbox/master/guide/.vuepress/public/logo_small.png"  width="100"/></a> | [Foolbox](https://github.com/bethgelab/foolbox) | Foolbox is a Python library that lets you easily run adversarial attacks against machine learning models like deep neural networks.    | ![GitHub stars](https://img.shields.io/github/stars/bethgelab/foolbox?style=social) | <a href="https://github.com/bethgelab/foolbox/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bethgelab/foolbox" height="15"/></a> |













