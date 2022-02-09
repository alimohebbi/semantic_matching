# Setup

- virtualenv
- python3.7
- python3.7-dev
- g++
- build-essential
- libssl-dev
- 32 GB RAM 

> **Note:** The 32GB RAM requirement is needed for the fast text word embedding models to be used. The rest of the models can be used with 16GB.

The requirements can be installed with the following command.

```sh
sudo apt install virtualenv  python3.7 python3.7-dev build-essential libssl-dev
```

1. Create a virtual environment:
```sh
virtualenv -p /usr/bin/python3.7 venv
```

2. Activate the environment:
```sh
source venv/bin/activate
```

3. Install required packages

```shell
pip install -r requirements.txt
```

## Run
1. Modify `config.yml` following entry:
 - `model_dir` : path to the word embedding models with respect to `model_path` entry

1. Run semantic matching

```shell
python run_all_combinations.py
```

2. check the results
    - MRR and top1 values in available the `final.csv`.
    - Results of the table in the paper are available in `table_mrr.csv` and `tabel_top1.csv`
