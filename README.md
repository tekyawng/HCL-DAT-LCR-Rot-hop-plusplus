# Hierarchical Contrastive Learning Approach for Aspect-Based Sentiment Classification
## Project Setup Instructions

1. **Install Python 3.7**: Make sure to install Python 3.7. Newer versions are not compatible with the required packages. Python can be downloaded from [this link](https://www.python.org/downloads/release/python-370/)

3. **Install Anaconda**: Download and install Anaconda from [this link](https://www.anaconda.com/products/distribution).

---

## Virtual Environment Setup

### 1. Create a Virtual Environment:

- Open Anaconda and create a virtual environment using Python 3.7:

  ```bash
  conda create --name name_of_env python=3.7
  ```

- Activate the environment:

  ```bash
  conda activate name_of_env
  ```

### 2. Download the zip folder, extract it and navigate to the project folder:

- Navigate to the folder

  ```bash
  cd repo_folder_path
  ```

### 3. Install Required Packages:

- Run the following command to install required packages (including `protobuf 3.19` and `tensorflow 1.15`):

  ```bash
  pip install -r requirements.txt
  ```

## Using and Running Code
1. **Generate Raw Data**: Run `raw_data.py` to obtain the raw data for your required domains (restaurant, laptop, and book).

2. **(Optional) Get BERT embeddings**: The BERT embeddings have been pushed to this repository and therefore need not be generated again, but if you deem to generate them: Run `bert_prepare.py` to obtain the raw data for your required domains (restaurant, laptop, and book). Place the files in the embedding directories, and rename them if necessary.

3. **Tune hyperparameters**: Run `main_hyper.py` to find the optimal hyperparameter settings. The number of configurations and iterations can be changed here. `main_test.py` already contains the optimal hyperparameters for the all models as presented in the paper.

4. **Adjust additional settings**: Changing settings in `config.py`, `main_test.py` or `main_hyper.py` allows for running the model with other settings for e.g. epochs, adding or leaving out neutral sentiment, etc.

5. **Adjust discriminator structure**: `nn_layer.py` can be used to change the structure of the discriminators and refinment FFNs.

6. **Run the model**: Fill `main_test.py` with the hyperparameters of choice and run the model for a given amount of epochs. Results will be stored in Result_Files, including runtime, accuracy per sentiment polarity, train accuracy and general (maximum) test accuracy. t-SNE plots will be stored in the t-SNE folder. Uncomment the models you want to train and test to obtain results, using the according hyperparameters.

**In case you have any questions regarding the use of the code, do not hesitate to contact me via 617509tn@student.eur.nl

## References

This code is adapted from Verschoor (2025).

[https://github.com/Johan-Verschoor/CL-XD-ABSA](https://github.com/Johan-Verschoor/CL-XD-ABSA).

Their code is based on the work of Knoester, Frasincar and Trușca (2022)

[https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS/](https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS/)

Knoester, J., Frasincar, F., and Trușca, M. M. (2022). Domain adversarial training for aspect-based sentiment analysis.  
In *22nd International Conference on Web Information Systems Engineering (WISE 2022)*, volume 13724 of LNCS, pages 21–37. Springer.

The work of Knoester et al. is an extension on the work of Trușca, Wassenberg, Frasincar and Dekker (2020).

[https://github.com/mtrusca/HAABSA_PLUS_PLUS](https://github.com/mtrusca/HAABSA_PLUS_PLUS)

Trușca M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using Deep Contextual Word Embeddings and Hierarchical Attention.  
In: *20th International Conference on Web Engineering (ICWE 2020)*. LNCS, vol 12128, pp. 365–380. Springer, Cham.  
[https://doi.org/10.1007/978-3-030-50578-3_25](https://doi.org/10.1007/978-3-030-50578-3_25)
