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

import pytest
import requests
import json


def test_multiple_paragraphs():
    model_endpoint = 'http://localhost:5000/model/predict'
    json_data = {
                    "paragraphs": [
                        {
                            "context": "John lives in Brussels and works for the EU",
                            "questions": [
                                "Where does John Live?",
                                "What does John do?",
                                "What is his name?"
                            ]
                        },
                        {
                            "context": "Jane lives in Paris and works for the UN",
                            "questions": [
                                "Where does Jane Live?",
                                "What does Jane do?"
                            ]
                        }
                    ]
                    }
    r = requests.post(url=model_endpoint, json=json_data)
    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'samples/small-dev.json'

    with open(file_path, 'rb') as file:
        json_data = json.load(file)
        r = requests.post(url=model_endpoint, json=json_data)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'

    with open(file_path) as file:
        q_objs = json.load(file)["paragraphs"][0]["questions"]

        # make sure all the correct questions have been returned
        all_questions = [q for q in q_objs]
        all_responses = response["predictions"][0]

        assert len(all_questions) == len(all_responses)

        # make sure answers are nonempty
        # note that this is not expected. However, for the questions in samples/small-dev, it is reasonable to
        # expect this model to at least provide an answer
        assert "" not in all_responses


if __name__ == '__main__':
    pytest.main([__file__])
