# MedicalTranscriptionClassification
This project attempts to classify the medical specialty of medical transcriptions. 

## Description
This project includes data exploration of the dataset, a support vector machine as the baseline model and RoBERTa as the pre-trained model to classify medical transcriptions.

## Getting Started

### Dependencies

* The Medical Transcription dataset is required to run the notebooks. [Medical Transcription Dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions)
* TensorFlow, Pandas, Spacy, Seaborn, SciKit Learn, Numpy, Transformers and MatPlotLib must be installed before running the notebooks

### List of Files
* FreyaGrayPresentation.pdf - slides used for presentation.
* baseline_model.ipynb - Implements the SVC support vector machine from SciKit Learn as a baseline model for classification.
* medical_transcription_data_analysis.ipynb - Initial data exploration of the Medical Transcription dataset.
* mtsamples.csv - The Medical Transcription dataset.
* roberta.ipynb - Implements RoBERTa as a pre-trained model for classification.

## Authors
Freya Gray
[Freya's GitHub](https://github.com/freyakgray)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
Dataset: Medical Transcription
* [Kaggle Link](https://www.kaggle.com/tboyle10/medicaltranscriptions) 

Hugging Face guide for pre-trained models
* [link](https://huggingface.co/docs/transformers/training)

Text cleaning and lemmatization guide
* [link](https://medium.com/@sourenh94/tweets-sentiment-analysis-using-deep-transfer-learning-6cab7009986f)

Data exploration guides
* [link](https://towardsdatascience.com/exploratory-text-analysis-in-python-8cf42b758d9e)
* [link](https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools)
