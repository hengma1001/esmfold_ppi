import logging
import os
import sys
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor
from proxystore.store import register_store
from proxystore.store.file import FileStore
from pydantic import Field

from esmfold_ppi.parsl import ComputeSettingsTypes
from esmfold_ppi.utils import (
    BaseModel,
    ProteinConfig,
    mkdir_validator,
    parse_seqs,
    path_validator,
)

# label = os.path.basename(pdb_file)[:-4]
# return {"name": label, "sequence": comp_seq}


def run_esmfold(
    protein: ProteinConfig, output_dir: Path, torch_hub_dir: Optional[str] = None
) -> ProteinConfig:
    """Run ESMFold on a sequence to predict structure.

    Parameters
    ----------
    pi : Dict[str, Any]
        The protein to fold, used to construct ProteinInteractions.
    output_dir : Path
        The path to write the output PDB file.

    Returns
    -------
    Dict[str, Any]
        The updated protein interactions with the ESMFold results.
    """
    print("In ESM fold task", flush=True)

    import MDAnalysis as mda

    from esmfold_ppi.esmfold import EsmFold

    esmfold = EsmFold(torch_hub_dir=torch_hub_dir)

    # Create the output directory and make the parent directory
    pdb_dir = output_dir / "pdb"
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    pdb_dir.mkdir(exist_ok=True, parents=True)

    # Create the output path
    filename = protein.id + ".pdb"
    output_pdb = pdb_dir / filename

    print(f"Running ESMFold model and outputting to {output_pdb}", flush=True)
    # Run the ESMFold model to predict structure
    ppi_structure = esmfold.run(protein.sequence)
    with open(output_pdb, "w") as f:
        f.write(ppi_structure)

    protein.pdb_file = output_pdb

    print("Calculating pLDDT score", flush=True)
    # Calculate the pLDDT score
    mda_u = mda.Universe(output_pdb)
    plddt = mda_u.select_atoms("protein and name CA").bfactors.mean()
    protein.plddt = plddt

    # Write the data to a JSON file
    filename = protein.id + ".json"
    protein.write_yaml(data_dir / filename)

    return protein


class ResultLogger:
    def __init__(self, result_dir: Path) -> None:
        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir

    def log(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)


class Thinker(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        df: pd.DataFrame,
        n_workers: int,
        result_logger: ResultLogger,
        **kwargs: Any,
    ) -> None:
        """The workflow.

        - It runs ESMFold for protein complex structure predict

        Parameters
        ----------
        sequence : List[str]
            The list of protein sequences to generate interactions for.
        result_logger : ResultLogger
            The result logger to use.
        """
        super().__init__(**kwargs)

        self.df = df
        self.n_workers = n_workers
        self.result_logger = result_logger
        self.task_idx = 0
        print(f"Running esmfold {len(self.df)} sequences. ")

    def submit_esmfold_task(self):
        if self.task_idx >= len(self.df):
            self.done.set()
            return

        input_dict = self.df.iloc[self.task_idx].to_dict()

        self.queues.send_inputs(
            ProteinConfig(**input_dict),
            method="run_esmfold",
            topic="esmfold",
            keep_inputs=False,
        )
        self.task_idx += 1

    @agent(startup=True)  # type: ignore[misc]
    def start_tasks(self) -> None:
        """On start up, submit all the generation tasks at once."""
        for _ in range(self.n_workers):
            self.submit_esmfold_task()

    @result_processor(topic="esmfold")
    def process_esm_result(self, result: Result):
        self.result_logger.log(result, "esmfold")
        print(result.value.dict())
        if not result.success:
            logging.warning(f"Bad inference result: {result.json()}")

        # The old task is finished, start a new one
        self.submit_esmfold_task()


class EsmFoldConfig(BaseModel):
    """The configuration for the ESMFold task."""

    torch_hub_dir: Optional[str] = Field(
        default=None,
        description="The path to the torch hub directory.",
    )


class WorkflowConfig(BaseModel):
    """Provide a YAML interface to configure the workflow."""

    # Workflow parameters
    host_fa: Path = Field(..., description="Path to the host fasta file")
    viral_fa: Path = Field(..., description="Path to the viral fast file")
    output_dir: Path = Field(..., description="Path to the workflow output directory.")

    esmfold: EsmFoldConfig = Field(..., description="ESMFold config.")
    small_subset: int = Field(
        default=0,
        description="The number of proteins to use for testing the workflow.",
    )
    compute_settings: ComputeSettingsTypes = Field(
        ..., description="The compute settings to use."
    )

    # validators
    _output_dir_mkdir = mkdir_validator("output_dir")
    _hfa_file_exists = path_validator("host_fa")
    _vfa_file_exists = path_validator("viral_fa")

    def configure_logging(self) -> None:
        """Set up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.output_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = WorkflowConfig.from_yaml(args.config)
    cfg.write_yaml(cfg.output_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.output_dir / "proxy-store"))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["esmfold"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the get_config
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.get_config(cfg.output_dir / "run-info")

    # Assign constant settings to the esmfold task function
    my_run_esmfold = partial(
        run_esmfold, output_dir=cfg.output_dir / "esmfold", **cfg.esmfold.dict()
    )
    update_wrapper(my_run_esmfold, run_esmfold)

    doer = ParslTaskServer([my_run_esmfold], queues, parsl_config)

    # Create the result logger
    result_logger = ResultLogger(cfg.output_dir / "result")

    logging.info("Loading proteins")

    # Load the protein interaction data
    host_seqs = parse_seqs(cfg.host_fa, seq_type="host")
    viral_seqs = parse_seqs(cfg.viral_fa, seq_type="viral")

    seq_df = pd.DataFrame(host_seqs + viral_seqs)
    seq_df.to_pickle(cfg.output_dir / "seq.pkl")
    # Use a small subset of the proteins for testing
    if cfg.small_subset:
        seq_df = seq_df[: cfg.small_subset]
    # proteins = [get_comp_seq(pdb) for pdb in tqdm(sequences)]

    logging.info(f"Loaded {len(seq_df)} proteins")

    n_workers = cfg.compute_settings.available_accelerators
    n_workers = len(n_workers) if isinstance(n_workers, list) else int(n_workers)
    thinker = Thinker(
        queue=queues,
        df=seq_df,
        n_workers=n_workers,
        result_logger=result_logger,
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        # Send the kill signal to the task server
        queues.send_kill_signal()

        # Wait for the task server to complete
        doer.join()

        # Clean up proxy store
        store.close()

    logging.info("Workflow has completed")
