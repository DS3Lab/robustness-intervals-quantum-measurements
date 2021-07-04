# Robust Intervals for Expectation Values
### Basic Usage
To run VQE, run the following command: 
```
python run_vqe.py --molecule <mol_name> --ansatz <ansatz_name> --backend <backend_name>
```
where `mol_name` is one of `h2, lih, beh2`, `ansatz_name` is one of `upccgsd, spa, spa-s, spa-gs, spa-gas` and 
`backend_name` specifies which quantum simulator is used by tequila.

To compute robustness intervals, run:
```
python compute_intervals.py --molecule <mol_name> --results_dir <path to results> --use_grouping <0 or 1>
```
