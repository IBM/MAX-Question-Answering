#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restplus import fields
from flask_restplus import abort

# Set up parser for input data
# (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
context_example = 'John lives in Brussels and works for the EU'
question_example = 'Where does John Live?'
id_example = '1'
paragraphs_example = [
    {
      "context": "John lives in Brussels and works for the EU",
      "questions": [
        "Where does John Live?", "What does John do?", "What is his name?"
      ]
    },
    {
      "context": "Jane lives in Paris and works for the UN",
      "questions": [
        "Where does Jane Live?", "What does Jane do?"
      ]
    }
  ]

article = MAX_API.model('Article JSON object', {
    'context': fields.String(required=True, description="Text where answers to questions can be found.",
                             example=context_example),
    'questions': fields.List(fields.String(required=True, description="Questions to be answered from the context.",
                                           example=question_example))
})

input_parser = MAX_API.model('Data JSON object', {
    'paragraphs': fields.List(fields.Nested(article),
                              description="List of paragraphs, each with a context and follow up questions.",
                              example=paragraphs_example)
})

# Creating a JSON response model:
# https://flask-restplus.readthedocs.io/en/stable/marshalling.html#the-api-model-factory

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.List(fields.String), description='Predicted answers to questions')
})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser, validate=True)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        input_json = MAX_API.payload
        try:
            for p in input_json["paragraphs"]:
                if not frozenset(p.keys()) == frozenset(["context", "questions"]):
                    abort(400, "Invalid input, please provide a context and questions.")
                if p["context"] == "":
                    abort(400, "Invalid input, please provide a paragraph.")
                if not isinstance(p["questions"], list):
                    abort(400, "Invalid input, questions should be a list.")
        except KeyError:
            abort(400, "Invalid input, please check that the input JSON has a `paragraphs` field.")
        except AssertionError:
            abort(400, "Invalid input, please ensure that the input JSON has `context` and `questions` fields.")

        preds = self.model_wrapper.predict(input_json)
        # Create a flat list of answers
        answers_list = ["" if not preds[p][0] else preds[p][1] for p in preds]
        # Create a split of how many elements go in each list
        splits = [len(p['questions']) for p in input_json['paragraphs']]
        # Create an empty answers list
        answers = []
        # Populate this list based on above split of nested lists
        for i, s in enumerate(splits):
            if i == 0:
                answers.append(answers_list[0:s])
            else:
                answers.append(answers_list[splits[i-1]:splits[i-1]+s])

        result['predictions'] = answers
        result['status'] = 'ok'

        return result
