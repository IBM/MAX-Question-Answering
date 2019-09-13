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
# Also contains code from
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").

from maxfw.model import MAXModelWrapper
import collections
import logging
from config import DEFAULT_MODEL_PATH, API_DESC, API_TITLE
from core.run_squad import read_squad_examples, convert_examples_to_features
from core.tokenization import FullTokenizer, BasicTokenizer
import tensorflow as tf
import numpy as np
from tensorflow.contrib import predictor
import six

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': 'max-question-answering',
        'name': API_TITLE,
        'description': API_DESC,
        'type': 'Natural Language Processing',
        'source': 'https://developer.ibm.com/exchanges/models/all/max-question-answering/',
        'license': 'Apache 2.0'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # Parameters for inference (need to be the same values the model was trained with)
        self.max_seq_length = 512
        self.doc_stride = 128
        self.max_query_length = 64
        self.max_answer_length = 30

        # Initialize the tokenizer
        self.tokenizer = FullTokenizer(
            vocab_file='assets/vocab.txt', do_lower_case=True)

        self.predict_fn = predictor.from_saved_model(DEFAULT_MODEL_PATH)

        logger.info('Loaded model')

    def _pre_process(self, inp):
        # if question ids are not included, generate them
        # Note: this may not work if the input data only has question ids for some of the questions
        unique_id = 1
        for article in inp["paragraphs"]:
            questions = article["questions"]
            for i in range(len(questions)):
                try:
                    questions[i]["id"]
                    continue
                except Exception:
                    new_question = {"id": str(unique_id),
                                    "question": questions[i]}
                    article["questions"][i] = new_question
                    unique_id += 1

        # convert answers to input features
        predict_examples = read_squad_examples(inp)
        features = convert_examples_to_features(predict_examples,
                                                self.tokenizer, self.max_seq_length,
                                                self.doc_stride, self.max_query_length)

        return features, predict_examples

    def _post_process(self, result):
        # convert to text predictions
        all_results = result[0]
        all_features = result[1][0]
        predict_examples = result[1][1]

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(predict_examples):
            features = example_index_to_features[example_index]
            prelim_preds = []
            feature = features[0]
            result = unique_id_to_result[feature.unique_id]
            start_indices = self._get_best_indices(result.start_logits, 10)
            end_indices = self._get_best_indices(result.end_logits, 10)

            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions within the 10 best predictions.
            for start_index in start_indices:
                for end_index in end_indices:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length >= self.max_answer_length:
                        continue
                    prelim_preds.append(_PrelimPrediction(
                        feature_index=0,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

            # use best prediction
            pred = None
            if len(prelim_preds) == 0:
                pred = _PrelimPrediction(
                    feature_index=0,
                    start_index=0,
                    end_index=0,
                    start_logit=0,
                    end_logit=0)
            else:
                pred = prelim_preds[0]

            final_text = ""
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[
                    pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[
                    orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = self.get_final_text(tok_text, orig_text, True)

            all_predictions[example.qas_id] = (example.question_text, final_text)

        return all_predictions

    def _predict(self, x, batch_size=32):
        features = x[0]

        predictions = []
        for i in range(0, len(features)):
            result = self.predict_fn({
                "unique_ids": np.array(features[i].unique_id).reshape(1),
                "input_ids": np.array(features[i].input_ids).reshape(-1, self.max_seq_length),
                "input_mask": np.array(features[i].input_mask).reshape(-1, self.max_seq_length),
                "segment_ids": np.array(features[i].segment_ids).reshape(-1, self.max_seq_length)
            })

            predictions.append(result)

        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

        all_results = []
        for result in predictions:
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        return all_results, x

    def _get_best_indices(self, logits, n_best_size):
        """Get the best logits from a list."""
        index_and_score = sorted(
            enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return ns_text, ns_to_s_map

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            tf.logging.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            tf.logging.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text
