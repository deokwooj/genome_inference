"""Testing for GenomeSequenceGenerator and  GenomeSequenceInference"""

import unittest

import numpy as np
import pandas as pd
from numpy import testing as np_test
from scipy.stats import bernoulli

from inference.data_generator import GenomeSequenceGenerator, flip_err
from inference.inference import GenomeSequenceInference, df2counter


def flip_err_gth(seq_gth: np.ndarray, seq_01: np.ndarray, seq_10: np.ndarray) -> np.ndarray:
    """
    brute-force flipping to get the ground truth of bit flips

    Parameters
    ----------
    seq_gth : original boolean sequence.
    seq_01 : index array to toggle from 0 to 1.
    seq_10 : index array to toggle from 1 to 0.

    Returns
    -------
    flipped boolean sequence.

    """

    seq_obs2 = np.nan * np.zeros_like(seq_gth, dtype=bool)
    for i in range(len(seq_gth)):
        gth_state = seq_gth[i]
        if gth_state:  # True state
            if seq_10[i]:
                seq_obs2[i] = ~seq_gth[i]  # Flip
            else:
                seq_obs2[i] = seq_gth[i]
        else:  # False state
            if seq_01[i]:
                seq_obs2[i] = ~seq_gth[i]  # Flip
            else:
                seq_obs2[i] = seq_gth[i]

    assert np.isnan(seq_obs2.sum()) == False
    return seq_obs2.astype(bool)


class GenomeInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p1 = 0.4
        self.p01 = 0.1
        self.p10 = 0.2
        self.seq_len = 5
        self.max_read_len = 100
        self.num_reads = 100
        self.num_tests = 10
        self.seq_sim = GenomeSequenceGenerator(p1=self.p1,
                                               p01=self.p01,
                                               p10=self.p10,
                                               max_read_len=self.max_read_len)

    def test_flip_err(self, ):
        """
        Test bit-flipping
        """
        for i in range(self.num_tests):
            bern_rv_1 = bernoulli(self.p1)
            bern_rv_10 = bernoulli(self.p10)
            bern_rv_01 = bernoulli(self.p01)

            seq_gth = bern_rv_1.rvs(self.seq_len).astype(bool)  # ground truth sequence.
            seq_01 = bern_rv_01.rvs(self.seq_len).astype(bool)  # ground truth sequence.
            seq_10 = bern_rv_10.rvs(self.seq_len).astype(bool)  # ground truth sequence.
            seq_obs1 = flip_err(seq_gth, seq_01, seq_10)
            seq_obs2 = flip_err_gth(seq_gth, seq_01, seq_10)
            assert all(seq_obs1 == seq_obs2)

    def test_inference(self, ):
        """
        Test inference algorithm

        """
        for i in range(self.num_tests):
            data_seq_gth = self.seq_sim.get_simul_data(self.seq_len)
            df_test_data = self.seq_sim.reading_genome_seq(data_seq_gth, self.num_reads)
            sample_cnt = df2counter(df_test_data)
            seq_infer = GenomeSequenceInference(p1=self.p1, p01=self.p01, p10=self.p10)
            prob1 = seq_infer.compute_posterior_prob(seq_len=max(sample_cnt.keys()) + 1,
                                                     sample_cnt=sample_cnt)
            df_res = pd.DataFrame(data=data_seq_gth.astype(int), columns=['gth'])
            df_res['prob'] = prob1
            assert df_res.apply(lambda x: abs(round(x[1]) - x[0]), axis=1).sum() == 0

    def test_generator(self, ):
        """
        Test data generator
        """
        p1, p01, p10 = 0.45, 0.45, 0.45
        seq_len, max_read_len, num_reads = 100000, 10000, 10
        seq_sim = GenomeSequenceGenerator(p1=p1, p01=p01, p10=p10, max_read_len=max_read_len)

        data_seq_gth = seq_sim.get_simul_data(seq_len).astype(int)
        p1_empr = np.count_nonzero(data_seq_gth) / seq_len
        np_test.assert_array_almost_equal(p1_empr, p1, decimal=2)

        df_test_data = seq_sim.reading_genome_seq(seq_gth=data_seq_gth,
                                                  num_reads=num_reads,
                                                  to_csv=True,
                                                  read_idx_s=0,
                                                  read_len=seq_len)

        for idx_s, read_str in df_test_data.to_numpy():
            gth_blk = data_seq_gth[idx_s:idx_s + len(read_str)]
            read_blk = np.array([int(s) for s in read_str])
            assert len(gth_blk) == len(read_blk)
            one_idx = gth_blk == 1
            n_ones = np.count_nonzero(one_idx)
            n_zeros = len(gth_blk) - n_ones
            n_one_flips = n_ones - np.count_nonzero(read_blk[one_idx])
            n_zeros_flips = np.count_nonzero(read_blk[~one_idx])
            p10_empr = n_one_flips / n_ones
            p01_empr = n_zeros_flips / n_zeros
            np_test.assert_array_almost_equal(p10_empr, p10, decimal=2)
            np_test.assert_array_almost_equal(p01_empr, p01, decimal=2)

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
