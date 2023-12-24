import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml
from Bio import SeqIO
from pydantic import BaseModel as _BaseModel
from pydantic import Field, validator
from tqdm import tqdm

T = TypeVar("T")
PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    """Check if a path exists (implements path_validator)."""
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p.absolute()


def _resolve_mkdir(value: Path) -> Path:
    """Create a directory if it does not exist (implements mkdir_validator)."""
    p = value.resolve()
    p.mkdir(exist_ok=False, parents=True)
    return p.absolute()


def path_validator(field: str) -> classmethod:
    """Pydantic validator to check if a path exists."""
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def mkdir_validator(field: str) -> classmethod:
    """Pydantic validator to create a directory if it does not exist."""
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_mkdir)
    return _validator


class BaseModel(_BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models"""

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, "w") as fp:
            json.dump(self.dict(), fp, indent=2)

    @classmethod
    def from_json(cls: Type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore


class ProteinConfig(BaseModel):
    id: str
    sequence: str
    type: str
    pdb_file: Path = Field(None)
    plddt: float = Field(0.0)


def get_accID(prot_name):
    return prot_name.split("|")[1]


def parse_seqs(fasta_file, seq_type=None):
    records = list(SeqIO.parse(fasta_file, "fasta"))

    seq_info = []
    for record in tqdm(records):
        seq = str(record.seq)
        name = get_accID(record.name)
        local_df = {
            "id": name,
            "sequence": seq,
            "type": seq_type,
        }
        seq_info.append(local_df)
    return seq_info
