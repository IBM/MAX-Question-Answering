## How to train this model using your own data

- [Prepare your data for training](#prepare-data-for-training)
- [Train the model](#train-the-model)
 - [Rebuild the model-serving microservice](#rebuild-the-model-serving-microservice) 


## Prepare Data for Training

To prepare your data for training complete the steps listed in [data_preparation/README.md](data_preparation/README.md).

## Train the model


In this document `$MODEL_REPO_HOME_DIR` refers to the cloned MAX model repository directory, e.g.
`/users/hi_there/MAX-Question-Answering`. 

### Install local prerequisites

Open a terminal window, change dir into `$MODEL_REPO_HOME_DIR/training` and install the Python prerequisites. (Model training requires Python 3.6 or above.)

   ```
   $ cd training/

   $ pip install -r requirements.txt
    ... 
   ```

### Run the setup script

1. Locate the training configuration file. It is named `max-question-answering-training-config.yaml`.

   ```

   $ ls *.yaml
     max-question-answering-training-config.yaml 
   ```

1. Configure your environment for model training.

   ```
    $ python wml_setup.py max-question-answering-training-config.yaml
     ...
   ```
   
1. Once setup is completed, define the displayed environment variables.

   MacOS example:

   ```
   $ export ML_INSTANCE=...
   $ export ML_APIKEY=...
   $ export ML_ENV=...
   $ export AWS_ACCESS_KEY_ID=...
   $ export AWS_SECRET_ACCESS_KEY=...
   ```

### Train the model using Watson Machine Learning

1. Verify that the training preparation steps complete successfully.

   ```
    $ python wml_train.py max-question-answering-training-config.yaml prepare
     ...
     # --------------------------------------------------------
     # Checking environment variables ...
     # --------------------------------------------------------
     ...
   ```

   If preparation completed successfully:

    - Training data is present in the Cloud Object Storage bucket that WML will access during model training.
    - Model training code is packaged `max-question-answering-model-building-code.zip`

1. Start model training.

   ```
   $ python wml_train.py max-question-answering-training-config.yaml
    ...
    # --------------------------------------------------------
    # Starting model training ...
    # --------------------------------------------------------
    Training configuration summary:
    Training run name     : train-max-question-answering
    Training data bucket  : ...
    Results bucket        : ...
    Model-building archive: max-question-answering-model-building-code.zip
    Model training was started. Training id: model-...
    ...
   ```

1. Monitor training progress

   ```
   ...
   Training status is updated every 15 seconds - (p)ending (r)unning (e)rror (c)ompleted: 
   ppppprrrrrrr...
   ```

   After training has completed the training log file `training-log.txt` is downloaded along with the trained model artifacts.

   ```
   ...
   # --------------------------------------------------------
   # Downloading training log file "training-log.txt" ...
   # --------------------------------------------------------
   Downloading "training-.../training-log.txt" from bucket "..." to "training_output/training-log.txt"
   ..
   # --------------------------------------------------------
   # Downloading trained model archive "model_training_output.tar.gz" ...
   # --------------------------------------------------------
   Downloading "training-.../model_training_output.tar.gz" from bucket "..." to "training_output/model_training_output.tar.gz"
   ....................................................................................
   ```

   > If training was terminated early due to an error only the log file is downloaded. Inspect it to identify the problem.

   ```
   $ ls training_output/
     model_training_output.tar.gz
     trained_model/
     training-log.txt 
   ```

1. Return to the parent directory

### Rebuild the model-serving microservice

The model-serving microservice out of the box serves the pre-trained model. To serve your trained model you have to rebuild the Docker image:

Rebuild the Docker image

   ```
   $ docker build -t max-question-answering --build-arg use_pre_trained_model=false .
   ```
   
   > If the optional parameter `use_pre_trained_model` is set to `true` or if the parameter is not defined the Docker image will be configured to serve the pre-trained model.
   
 Once the Docker image build completes you can start the microservice as usual:
 
 ```
 $ docker run -it -p 5000:5000 max-question-answering
 ```
