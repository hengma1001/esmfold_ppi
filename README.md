# esm_ppi

<!-- [![docs](https://github.com/gpauloski/python-template/actions/workflows/docs.yml/badge.svg)](https://github.com/gpauloski/python-template/actions)
[![tests](https://github.com/gpauloski/python-template/actions/workflows/tests.yml/badge.svg)](https://github.com/gpauloski/python-template/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gpauloski/python-template/main.svg)](https://results.pre-commit.ci/latest/github/gpauloski/python-template/main) -->

The package is built to fold single protein and protein-protein complexes using ESMFold. 

Built on [python template](https://github.com/gpauloski/python-template/)

<!-- ## Setup Instructions

1. Click the "Use this template" button at the top right of this page.
2. Delete and directories you will not be using (commonly `docs/` if you do not want to use MKDocs or `examples/` if you will not have example code).
3. Follow the instructions to create the new repo then clone your repo locally.
4. The template uses "foobar" to indicate things that need to be changed.
   Start by searching for all instances (`git grep foobar`) and changing them accordingly.
5. Configure pre-commit:
    - Go to [https://pre-commit.ci/](https://pre-commit.ci/) and enable pre-commit on your repo.
    - Update the pre-commit badge URL in this README with your new badge URL.
6. Configure GitHub pages:
    - Go to the "Pages" section of your repository settings.
    - Select "Deploy from a branch" and use the "gh-pages" branch.
7. Configure PyPI releases (if relevant):
    - Create a new API token for [https://pypi.org/](https://pypi.org/).
    - Add the token as a GitHub actions secret (see the instructions [here](https://github.com/pypa/gh-action-pypi-publish)).
8. Delete this boilerplate stuff in the README.
9. Commit and push changes.

### GitHub Configuration

I recommend making a few other changes to the repo's setting on GitHub.
- In "General"
  - Select/deselect features you need/don't need.
  - Select "Automatically delete head branches
- In "Branches": enable branch protection on `main`.
  - Check "Require a pull request before merging"
  - Check "Require status checks to pass before merging"
    - Check "Require branches to be up to date before merging"
    - Set required checks (e.g., pre-commit.ci, tests, etc.)
  - Check "Do not allow bypassing the above settings" -->

## Installation

Clone the repo to your local machine. 

```bash
git clone https://github.com/hengma1001/esmfold_ppi.git
cd esmfold_ppi
```

Create a conda environment for the folding applications. 

```bash
conda create -n esmfold python=3.9
conda activate esmfold
```

Install PyTorch. 

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```

Install other dependencies. 

```bash
pip install -r envs/requirements.txt
```

Install the package. 

```bash 
pip install -e .
```

## Run the folding

The esmfold run can be initiated with following commands. 

```bash
cd examples
python -m esmfold_ppi.run_fold -c fold.yml
```

For the input yaml format, the variables are defined as follow. 

```yaml
# input sequences in fasta format
seq_file: ./example.fa
# output directory for the colmena workflow
output_dir: run_test/

esmfold:
  # the cache directory for the esmfold weight
  torch_hub_dir: ~/.cache

# compute setting for parsl
compute_settings:
  # use local machine
  name: workstation
  # number of GPUs to use, can also be a list of gpu ids 
  available_accelerators: 2
```

...
