import argparse

import pandas as pd

from inference import GenomeSequenceInference, read_csv, export_csv
from inference import df2counter


class Application(object):
    def __init__(self, priori1, p01, p10, file_in, file_out):
        self.priori1 = priori1
        self.p01 = p01
        self.p10 = p10
        self.file_in = file_in
        self.file_out = file_out
        self.df_test = None
        self.sample_cnt = None

    def run(self, ):
        print(f'run inference algorithm with p1={self.priori1}, p01={self.p01}, p10={self.p10} ...')
        self.seq_infer = GenomeSequenceInference(p1=self.priori1,
                                                 p01=self.p01,
                                                 p10=self.p10)

        self.df_test = read_csv(fname=self.file_in)
        self.sample_cnt = df2counter(self.df_test)
        self.seq_len = max(self.sample_cnt.keys()) + 1
        prob1 = self.seq_infer.compute_posterior_prob(seq_len=self.seq_len,
                                                      sample_cnt=self.sample_cnt)
        print(f'export inference result to {self.file_out} ...')
        export_csv(df=pd.DataFrame(prob1, columns=['prob1']).round(3),
                   fname=self.file_out,
                   index=True,
                   index_label='idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shame', action='store', dest='file_in', help='test data filename')
    parser.add_argument('--out', action='store', dest='file_out', help='inference out filename')
    parser.add_argument('--p01', action='store', dest='p01', help='0 to 1 error rate')
    parser.add_argument('--p10', action='store', dest='p10', help='1 to 0 error rate')
    parser.add_argument('--prior1', action='store', dest='p1', help='prior probability that each base is a 1')

    args = parser.parse_args()
    app = Application(priori1=float(args.p1),
                      p01=float(args.p01),
                      p10=float(args.p10),
                      file_in=str(args.file_in),
                      file_out=str(args.file_out))
    app.run()
