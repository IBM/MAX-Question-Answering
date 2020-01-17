## How to prepare your data for training

Follow the instructions in this document to prepare your data for model training.
- [Prerequisites](#prerequisites)
- [Preparing your data](#preparing-your-data)

## Prerequisites
1. Have the answers to the questions that are being used for training.
2. Make sure each article that is being used is associated with a certain set of questions.

## Preparing your data
1. Convert data into the format required by the model, the same format as [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) and the `sample-train.json` file.
The format is shown below for reference. The full example JSON can be found in the sample training [file](../../training/sample_training_data/sample-train.json).

```json
{
   "data":[
      {
         "title":"",
         "paragraphs":[
            {
               "context":"",
               "qas":[
                  {
                     "answers":[
                        {
                           "answer_start":"int",
                           "text":""
                        }
                     ],
                     "question":"",
                     "id":""
                  }
               ]
            }
         ]
      }
   ]
}
```
