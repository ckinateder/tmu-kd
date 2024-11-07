# Copyright (c) 2023 Ole-Christoffer Granmo
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import typing


from tmu.models.base import MultiClauseBankMixin, MultiWeightBankMixin, TMBaseModel
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.util.encoded_data_cache import DataEncoderCache
from tmu.util.statistics import MetricRecorder
from tmu.weight_bank import WeightBank
import numpy as np
import logging
import pdb
from typing import Tuple

_LOGGER = logging.getLogger(__name__)


class DistillationClassifier(TMClassifier):
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        confidence_driven_updating: bool = False,
        type_i_ii_ratio: float = 1,
        type_i_feedback: bool = True,
        type_ii_feedback: bool = True,
        type_iii_feedback: bool = False,
        d: float = 200,
        platform: str = "CPU",
        patch_dim=None,
        feature_negation=True,
        boost_true_positive_feedback=1,
        reuse_random_feedback=0,
        max_included_literals=None,
        number_of_state_bits_ta=8,
        number_of_state_bits_ind=8,
        weighted_clauses=False,
        clause_drop_p=0,
        literal_drop_p=0,
        batch_size=100,
        incremental=True,
        type_ia_ii_feedback_ratio=0,
        absorbing=-1,
        absorbing_include=None,
        absorbing_exclude=None,
        literal_sampling=1,
        feedback_rate_excluded_literals=1,
        literal_insertion_state=-1,
        seed=None,
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            confidence_driven_updating,
            type_i_ii_ratio,
            type_i_feedback,
            type_ii_feedback,
            type_iii_feedback,
            d,
            platform,
            patch_dim,
            feature_negation,
            boost_true_positive_feedback,
            reuse_random_feedback,
            max_included_literals,
            number_of_state_bits_ta,
            number_of_state_bits_ind,
            weighted_clauses,
            clause_drop_p,
            literal_drop_p,
            batch_size,
            incremental,
            type_ia_ii_feedback_ratio,
            absorbing,
            absorbing_include,
            absorbing_exclude,
            literal_sampling,
            feedback_rate_excluded_literals,
            literal_insertion_state,
            seed,
        )

    def _fit_sample(
            self,
            target: int,
            not_target: int | None,
            sample_idx: int,
            clause_active: np.ndarray,
            literal_active: np.ndarray,
            encoded_X_train
    ) -> Tuple[dict, np.ndarray, np.ndarray]:

        class_sum, clause_outputs = self.mechanism_clause_sum(
            target=target,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train,
            sample_idx=sample_idx
        )

        update_p_target: float = self._fit_sample_target(
            class_sum=class_sum,
            clause_outputs=clause_outputs,
            is_target_class=True,
            class_value=target,
            sample_idx=sample_idx,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train
        )

        # for incremental, and when we only have 1 sample, there is no other targets
        if not_target is None:
            return dict(
                update_p_target=update_p_target,
                update_p_not_target=None
            )

        class_sum_not, clause_outputs_not = self.mechanism_clause_sum(
            target=not_target,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train,
            sample_idx=sample_idx
        )

        update_p_not_target: float = self._fit_sample_target(
            class_sum=class_sum_not,
            clause_outputs=clause_outputs_not,
            is_target_class=False,
            class_value=not_target,
            sample_idx=sample_idx,
            clause_active=clause_active,
            literal_active=literal_active,
            encoded_X_train=encoded_X_train
        )

        return dict(
            update_p_not_target=update_p_not_target,
            update_p_target=update_p_target,
        ), class_sum, class_sum_not

    def fit(
            self,
            X: np.ndarray[np.uint32],
            Y: np.ndarray[np.uint32],
            shuffle: bool = True,
            metrics: typing.Optional[list] = None,
            teacher_sums: np.ndarray[np.float32] = None,
            teacher_sums_not: np.ndarray[np.float32] = None,
            teacher_multiplier: float = 1.0,
            *args,
            **kwargs
    )->Tuple[dict, np.ndarray, np.ndarray]:
        # both the teacher and student will export the same metrics and class sums
        # only if teacher sums are provided will the student be trained
        student = False
        if teacher_sums is not None:
            student = True
            assert teacher_sums_not is not None, "Teacher sums not provided"
            assert teacher_multiplier > 0, "Teacher multiplier must be greater than 0"

        metrics = metrics or []
        assert X.shape[0] == len(Y), "X and Y must have the same number of samples"
        assert len(X.shape) >= 2, "X must be a 2D array"
        assert len(Y.shape) == 1, "Y must be a 1D array"
        #assert X.dtype == np.uint32, "X must be of type uint32"
        #assert Y.dtype == np.uint32, "Y must be of type uint32"

        self.init(X, Y)
        self.metrics.clear()

        encoded_X_train: np.ndarray = self.train_encoder_cache.get_encoded_data(
            data=X,
            encoder_func=lambda x: self.clause_banks[0].prepare_X(x)
        )

        clause_active: np.ndarray = self.mechanism_clause_active()
        literal_active: np.ndarray = self.mechanism_literal_active()

        sample_indices: np.ndarray = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(sample_indices)

        class_sums = np.zeros(sample_indices.shape, dtype=np.float32)
        class_sums_not = np.zeros(sample_indices.shape, dtype=np.float32)

        for sample_idx in sample_indices:
            target: int = Y[sample_idx]
            not_target: int | None = self.weight_banks.sample(exclude=[target])

            history, class_sum, class_sum_not = self._fit_sample(
                target=target,
                not_target=not_target,
                sample_idx=sample_idx,
                clause_active=clause_active,
                literal_active=literal_active,
                encoded_X_train=encoded_X_train
            )

            # now update the class sums np.ndarray
            class_sums[sample_idx] = class_sum
            class_sums_not[sample_idx] = class_sum_not

            if "update_p" in metrics:
                self.metrics.add_scalar(group="update_p", key=target, value=history["update_p_target"])
                self.metrics.add_scalar(group="update_p", key=target, value=history["update_p_not_target"])

        return self.metrics.export(
            mean=True,
            std=True
        ), class_sums, class_sums_not
