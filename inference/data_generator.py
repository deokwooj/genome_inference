"""Genome sequence data simulator """
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import bernoulli

MaxReadLength = 1000
Test_Data_Fname = 'test_data.csv'
Gth_Data_Fame = 'gth_data.csv'


def flip_err(seq_gth, seq_01, seq_10):
    """
    Efficient bit flip function, 10x faster than a naive brute-force bit-flipping.

    Parameters
    ----------
    seq_gth : original boolean sequence.
    seq_01 : index array to toggle from 0 to 1
    seq_10 : index array to toggle from 1 to 0.

    Returns
    -------
    flipped boolean sequence.

    """

    seq_obs = seq_gth.copy()  # observed sequence
    seq_obs[seq_gth == False] ^= seq_01[seq_gth == False]
    seq_obs[seq_gth == True] ^= seq_10[seq_gth == True]
    return seq_obs


def read_csv(fname):
    df_test = pd.read_csv(fname, sep='\t', dtype=str)
    return df_test


def export_csv(df, fname, index, index_label):
    df.to_csv(fname, index=index, index_label=index_label, sep='\t')


class GenomeSequenceGenerator:
    def __init__(self, p1, p01, p10,
                 max_read_len=MaxReadLength,
                 test_data_fname=Test_Data_Fname,
                 gth_data_fname=Gth_Data_Fame):

        self.bern_rv_1 = bernoulli(p1)
        self.bern_rv_10 = bernoulli(p10)
        self.bern_rv_01 = bernoulli(p01)
        self.max_read_len = max_read_len
        self.test_data_fname = test_data_fname
        self.gth_data_fname = gth_data_fname

    def to_csv(self, df, fname):
        export_csv(df, fname, index=False, index_label='')

    def from_csv(self, data_type='test'):
        if data_type == 'test':
            fname = self.test_data_fname
        elif data_type == 'gth':
            fname = self.gth_data_fname
        else:
            raise Exception(ValueError, 'data_type must be [\'test\'|\'gth\']')
        return read_csv(fname)

    def get_simul_data(self, seq_len, to_csv=True):
        seq_gth = self.bern_rv_1.rvs(seq_len).astype(bool)  # ground truth sequence.

        if to_csv:
            df_gth_data = pd.DataFrame(seq_gth.astype(np.uint8), columns=['genome'])
            self.to_csv(df_gth_data, self.gth_data_fname)
            print(f'ground truth data is stored in {self.gth_data_fname}..')

        return seq_gth

    def _read_genome(self, seq_gth, read_idx_s=-1, read_len=-1):
        seq_len = len(seq_gth)

        if read_idx_s < 0:
            read_idx_s = np.random.randint(0, seq_len)  # a random starting point ~ [0, seq_len-1]

        if read_len < 0:
            read_len = np.random.randint(1, self.max_read_len + 1)  # a random reading length ~ [1, max_read_len]

        read_idx_t = min(read_idx_s + read_len, seq_len)  # a stopping point
        reading_blk = seq_gth[read_idx_s:read_idx_t]
        reading_blk_len = len(reading_blk)
        zero_bit_idx = reading_blk == 0
        n_zeros = np.count_nonzero(zero_bit_idx)
        n_ones = reading_blk_len - n_zeros
        seq_01 = np.zeros(reading_blk_len, dtype=bool)
        seq_10 = np.zeros(reading_blk_len, dtype=bool)
        seq_01[zero_bit_idx] = self.bern_rv_01.rvs(n_zeros).astype(bool)
        seq_10[~zero_bit_idx] = self.bern_rv_10.rvs(n_ones).astype(bool)
        seq_obs = flip_err(reading_blk, seq_01, seq_10).astype(np.uint8)
        return {'read_idx_s': read_idx_s, 'seq_read': ''.join(map(str, seq_obs))}

    def reading_genome_seq(self, seq_gth, num_reads, to_csv=True, read_idx_s=-1, read_len=-1):
        data_obs = defaultdict(list)
        for i in range(num_reads):
            read_out = self._read_genome(seq_gth, read_idx_s, read_len)
            data_obs['start_pos'].append(read_out['read_idx_s'])
            data_obs['data_read'].append(read_out['seq_read'])

        df_test_data = pd.DataFrame(data_obs)

        if to_csv:
            self.to_csv(df_test_data, self.test_data_fname)
            print(f'test data is stored in {self.test_data_fname} ..')

        return df_test_data


if __name__ == '__main__':
    """ Generate test sham files """
    import os

    export_dir = os.path.dirname(os.getcwd()) + os.sep
    p1, p01, p10 = 0.4, 0.1, 0.2
    seq_len, num_reads = 100000, 1000
    seq_sim = GenomeSequenceGenerator(p1=p1,
                                      p01=p01,
                                      p10=p10,
                                      max_read_len=MaxReadLength,
                                      test_data_fname=export_dir + 'test_data.shame',
                                      gth_data_fname=export_dir + 'gth_seq.shame')

    data_seq_gth = seq_sim.get_simul_data(seq_len, to_csv=True)

    df_test_data = seq_sim.reading_genome_seq(seq_gth=data_seq_gth,
                                              num_reads=num_reads,
                                              to_csv=True)
