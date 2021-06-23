# Robust Intervals for Expectation Values
### Basic Usage
To run basis set free VQE and calculate robustness intervals containing the true ground state 
energy, run the following command: 
```
python main.py --molecule <mol_name> --ansatz <ansatz_name> --backend <backend_name> --use_grouping <0 or 1>
```
where `mol_name` is one of `h2, lih, beh2`, `ansatz_name` is one of `upccgsd, spa, spa-s, spa-gs, spa-gas` and 
`backend_name` specifies which quantum simulator is used by tequila.

