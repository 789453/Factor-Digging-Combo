Set-StrictMode -Version 2
$env:PATH = "D:\Total_Tools\miniforge3\Scripts;" + $env:PATH
$env:CONDA_DEFAULT_ENV = "universal"
$env:CONDA_PREFIX = "D:\Total_Tools\miniforge3\envs\universal"

python d:\Trading\My_factor_mining_427\check_db.py