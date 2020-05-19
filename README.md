# genome-inference

genome-inference is a Python library for Bayesian genome-inference.

## Installation

### Dependencies


genome-inference requires:

- Python (>= 3.7)
- NumPy (>= 1.18.1)
- SciPy (>= 1.4.1)
- Pandas (>= 1.0.3)
- pytest (>= 5.0.1)

### Testing

  - For unit test, run  ```pytest inference ```

  - For demo test, run  ``` ./test.sh run```

## Usage
  - For data generation, go to /inference and run  ``` python data_generator.py```

  - For inference task, run 

  ```python nfer-app.py  --sham ${INFILE} --out ${OUTFILE} --p01  ${P01} --p10 ${P10}  --prior1 ${PRIOR1}```

## License
[BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause/)
