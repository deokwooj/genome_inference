"""Bayesian genome-inference algorithm"""

from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def df2counter(df_input: pd.DataFrame) -> defaultdict:
    """ Covert Pandas DataFrame into Counter Dictionary object. 

    Parameters
    ----------
    df_input : DataFrame object from test dataset.

    Returns
    -------
    Dictionary object such that Dict['idx']['val'] = 'cnt'
    which the number of occurance, 'cnt' of value ,'val' at bit position, 'idx'

    """
    sample_cnt = defaultdict(Counter)
    for (pos, read) in df_input.to_numpy():
        for i, s in enumerate(read):
            sample_cnt[i + int(pos)][int(s)] += 1
    return sample_cnt


class GenomeSequenceInference:
    """ Bayesian genome-inference algorithm

    Parameters
    ----------
    p1 : Prior probability that any given genomic base is a 1.
    p01 : Probability that a 0 base is measured as a 1.
    p10 : Probability that a 1 base is measured as a 0.
    """

    def __init__(self, p1: float, p01: float, p10: float):
        self.p1 = p1
        self.p0 = 1 - p1
        self.p01 = p01
        self.p00 = 1 - p01
        self.p10 = p10
        self.p11 = 1 - p10
        self.eps = np.finfo(float).eps
        self.log_pts = None

    def _get_log_pts(self, ):
        if self.log_pts is None:
            # add eps for numerical stability of matrix computatiob when p10 or p01 =0
            Pts = np.array([[self.p00 * self.p0 + self.eps, self.p01 * self.p0 + self.eps],
                            [self.p10 * self.p1 + self.eps, self.p11 * self.p1] + self.eps])
            self.log_pts = np.log(Pts)

        return self.log_pts

    def _compute_llh(self, n0, n1):
        log_Pts = self._get_log_pts()
        return log_Pts @ np.array([n0, n1]).T

    def compute_posterior_prob(self, seq_len, sample_cnt):
        """ Compute the posterior probability that the genome character is a 1 at each position.

        Parameters
        ----------
        seq_len : the length of genome sequence. 
        sample_cnt : A Counter Dictionary such that Dict['idx']['val'] = 'cnt'
        which the number of occurance, 'cnt' of value ,'val' at bit position, 'idx'

        Returns
        -------
        1d array of the posterior probability (softmax) for genome sequence.

        """
        prob_est_mat = self.p1 * np.ones(seq_len)
        for idx, cnt_set in sample_cnt.items():
            llh_prob = self._compute_llh(n0=cnt_set[0],
                                         n1=cnt_set[1])
            posterior_prob = np.exp(llh_prob)
            prob_est_mat[idx] = posterior_prob[1] / posterior_prob.sum()  # softmax posterior function

        return prob_est_mat


if __name__ == '__main__':
    from inference import GenomeSequenceGenerator

    p1 = 0.3
    p01 = 0.1
    p10 = 0.2
    max_read_len = 7
    num_reads = 50
    seq_len = 10

    seq_sim = GenomeSequenceGenerator(p1=p1, p01=p01, p10=p10, max_read_len=max_read_len)

    data_seq_gth = seq_sim.get_simul_data(seq_len, to_csv=False)
    df_test_data = seq_sim.reading_genome_seq(data_seq_gth, num_reads, to_csv=False)
    sample_cnt = df2counter(df_test_data)

    seq_infer = GenomeSequenceInference(p1=p1, p01=p01, p10=p10)
    prob1 = seq_infer.compute_posterior_prob(seq_len=max(sample_cnt.keys()) + 1,
                                             sample_cnt=sample_cnt)
    df_res = pd.DataFrame(data=data_seq_gth.astype(int), columns=['gth'])
    df_res['prob'] = prob1
    print(df_res.round(3))
