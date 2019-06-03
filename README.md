# Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization

We provide the source code for the paper **"[Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization](https://arxiv.org/abs/1808.06218)"**, accepted at EMNLP'18. If you find the code useful, please cite the following paper. 

    @inproceedings{lebanoff-song-liu:2018,
     Author = {Logan Lebanoff and Kaiqiang Song and Fei Liu},
     Title = {Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization},
     Booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
     Year = {2018}}


## Goal

* Our system seeks to summarize a set of articles (about 10) about the same topic.

* The code takes as input a text file containing a set of articles. See below on the input format of the files.


## Dependencies

The code is written in Python (v2.7) and TensorFlow (v1.4.1). We suggest the following environment:

* A Linux machine (Ubuntu) with GPU (Cuda 8.0)
* [Python (v2.7)](https://www.anaconda.com/download/)
* [TensorFlow (v1.4.1)](https://www.tensorflow.org/install/)
* [Pyrouge](https://pypi.org/project/pyrouge/)
* [NLTK](https://www.nltk.org/install.html)


## How to Generate Summaries

1. Clone this repo. Download this [ZIP](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view?usp=sharing) file containing the pretrained model from See et al. Move the folder `pretrained_model_tf1.2.1` into the `./logs/` directory.
    ```
    $ git clone https://github.com/ucfnlp/multidoc_summarization/
    $ mv pretrained_model_tf1.2.1.zip multidoc_summarization/logs
    $ cd multidoc_summarization/logs
    $ unzip pretrained_model_tf1.2.1.zip
    $ rm pretrained_model_tf1.2.1.zip
    $ cd ..
    ```

2. Format your data in the following way:

    One file for each topic. Distinct articles will be separated by one blank line (two carriage returns \n). Each sentence of the article will be on its own line. See `./example_custom_dataset/` for an example.

3. Convert your data to TensorFlow examples that can be fed to the PG-MMR model.
    ```
    $ python convert_data.py --dataset_name=example_custom_dataset --custom_dataset_path=./example_custom_dataset/
    ```

4. Run the testing script. This will create a file called `logs/tfidf_vectorizer/example_custom_dataset.dill`. If you change the contents of your dataset, you should delete this file so that the script will re-create the TF-IDF vectorizer which reflects the changes. The summary files are located in the `./logs/example_custom_dataset/decoded/` directory.
    ```
    $ python run_summarization.py --dataset_name=example_custom_dataset --pg_mmr
    ```

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We gratefully acknowledge the work of Abigail See whose [code](https://github.com/abisee/pointer-generator) was used as a basis for this project.

