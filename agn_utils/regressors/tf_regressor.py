import pathlib
from typing import Callable
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.estimator import BoostedTreesRegressor
from tensorflow.python.training.tracking.tracking import AutoTrackable

from .regressor import Regressor
from ..agn_logger import logger
from ..diagnostic_tools import timing


def make_input_fn(
    data: pd.DataFrame,
    labels: pd.Series,
    shuffle=True,
) -> Callable:
    def _input_fn() -> tf.data.Dataset:
        # dataset = tf.posteriors_list.Dataset.from_tensor_slices((posteriors_list.values, labels.values))
        # if shuffle:
        #     dataset = dataset.shuffle(len(posteriors_list))
        # return dataset
        return data.to_dict("list"), labels.values

    return _input_fn


class TfRegressor(Regressor):
    """
    https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesRegressor

    NOTE:
    ref the following do determine how to tune training hyper-params
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    """

    def __init__(
        self,
        input_parameters: List[str],
        output_parameters: List[str],
        outdir: str,
        model_hyper_param: Optional[Dict] = {},
    ):
        super().__init__(input_parameters, output_parameters, outdir)
        self.model_hyper_param = dict(
            n_batches_per_layer=1,
            model_dir=outdir,
            label_dimension=1,
            weight_column=None,
            n_trees=100,
            max_depth=6,
            learning_rate=0.1,
            l1_regularization=0.0,
            l2_regularization=0.0,
            tree_complexity=0.0,
            min_node_weight=0.0,
            config=None,
            center_bias=True,
            pruning_mode="none",
            quantile_sketch_epsilon=0.01,
            train_in_memory=True,
        )
        self.model_hyper_param.update(model_hyper_param)
        self.fc = [
            tf.feature_column.numeric_column(key=p)
            for p in self.input_parameters
        ]
        self.model = BoostedTreesRegressor(self.fc, **self.model_hyper_param)

    @timing
    def train(self, data: pd.DataFrame):
        super().train(data)
        train, test, train_labels, test_labels = self.train_test_split(data)
        train_fn = make_input_fn(train, train_labels)
        self.model.train(train_fn)
        logger.info("Training complete")
        self.test(test, test_labels)

    def test(self, data: pd.DataFrame, labels: pd.Series):
        test_fn = make_input_fn(data, labels, shuffle=False)
        result = self.model.evaluate(test_fn, steps=10)
        logger.info("MODEL TESTING:")
        for key, value in result.items():
            logger.info(f"\t {key} : {value}")

    def save(self):
        feature_spec = tf.feature_column.make_parse_example_spec(self.fc)
        serving_input_fn = (
            tf.estimator.export.build_parsing_serving_input_receiver_fn(
                feature_spec
            )
        )
        self.model.export_saved_model(self.savepath, serving_input_fn)

    def load(self):
        subdirs = [
            x
            for x in pathlib.Path(self.savepath).iterdir()
            if x.is_dir() and "temp" not in str(x)
        ]
        latest = str(sorted(subdirs)[-1])
        self.model = tf.saved_model.load(latest)

    def visualise(self):
        "https://mljar.com/blog/visualize-tree-from-random-forest/"
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame):
        if isinstance(self.model, AutoTrackable):

            def predict_in_fn(input_df):
                examples = []
                for index, row in input_df.iterrows():
                    feature = {}
                    for col, value in row.iteritems():
                        feature[col] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[value])
                        )
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    examples.append(example.SerializeToString())
                return tf.constant(examples)

            pred_fn = self.model.signatures["serving_default"]
            preds = pred_fn(predict_in_fn(data))
            preds = preds["outputs"].numpy().flatten()
        else:
            predict_in_fn = lambda: tf.data.Dataset.from_tensors(dict(data))
            pred_fn = self.model.predict
            preds = np.array(
                [p["predictions"][0] for p in pred_fn(predict_in_fn)]
            )
        return preds

    @property
    def n_trees(self) -> int:
        return self.model_hyper_param.get("n_trees")
