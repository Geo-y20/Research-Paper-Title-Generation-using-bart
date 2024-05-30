# ArXiv Title Prediction from Abstract Using BART

![ArXiv](https://storage.googleapis.com/kaggle-public-downloads/arXiv.JPG)

## Introduction

This repository contains a notebook that demonstrates how to use the [BART](https://arxiv.org/abs/1910.13461) Transformer model to perform title generation from abstracts. BART is a sequence-to-sequence model where both the input and targets are text sequences, commonly used for text summarization. In this project, the goal is to summarize titles from abstracts.

### Acknowledgements

This notebook borrows code from:
- [simpletransformers documentation](https://github.com/ThilinaRajapakse/simpletransformers)
- Andrew Lukyanenko's [Arxiv metadata exploration](https://www.kaggle.com/artgor/arxiv-metadata-exploration) kernel

## Libraries 📚⬇

The following libraries are used in this project:
- numpy
- pandas
- json
- gc
- re
- random
- tqdm
- sklearn
- matplotlib
- plotly
- seaborn
- warnings
- logging
- torch
- transformers
- tokenizers
- simpletransformers

## Data

The dataset used is the `arxiv-metadata-oai-snapshot.json` file from the ArXiv dataset. It includes metadata for research papers, such as titles, abstracts, categories, and authors.

## Preprocessing

The data is loaded using a generator function to prevent memory issues. The metadata is filtered to include papers published between 2010 and 2020. Categories are mapped to their full names for better readability.

## Model Training & Evaluation

The BART model is fine-tuned for title generation using the `simpletransformers` library. The training and evaluation datasets are created from the preprocessed data. The model is trained for 3 epochs with a batch size of 6.

## Inference

The trained model is used to generate titles from abstracts. The notebook includes examples of true titles versus predicted titles for evaluation.

## Visualization

The project includes visualizations to analyze the distribution of word counts in abstracts and titles. It also visualizes the top categories and authors over the years.

## Usage

To use the notebook, follow these steps:

1. Clone the repository.
    ```bash
    git clone https://github.com/your-username/arxiv-title-prediction.git
    ```
2. Install the required libraries.
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook.
    ```bash
    jupyter notebook arxiv-title-prediction-from-abstract-using-bart.ipynb
    ```

## Results

The model demonstrates the capability to generate titles from abstracts with reasonable accuracy. Further improvements can be made by tuning hyperparameters or using a larger dataset.

## Conclusion

This project showcases the application of the BART model for text summarization tasks, specifically generating research paper titles from abstracts. The approach can be extended to other text summarization tasks with appropriate adjustments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to open an issue or contact us at [georgeyouhana2@gmail.com](mailto:georgeyouhana2@gmail.com) or [ffathy2004@gmail.com](mailto:ffathy2004@gmail.com).
