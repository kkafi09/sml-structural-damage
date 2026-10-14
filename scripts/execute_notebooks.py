"""Execute all analysis notebooks in fresh kernels and persist real outputs."""

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(path):
    """Run one notebook with this interpreter, preserving outputs even on failure."""
    path = Path(path)
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        notebook, km=manager, timeout=None, allow_errors=False,
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )

    def completed(cell, cell_index, execute_reply):
        """Save each completed cell so long-running executions remain inspectable."""
        nbformat.write(notebook, path)
        print(f"{path.name}: cell {cell_index + 1} completed", flush=True)

    client.on_cell_executed = completed
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"Finished {path.name}", flush=True)


def main():
    """Run the three notebooks in order, or selected notebook filenames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", default=["01_eda.ipynb", "02_modeling.ipynb", "03_submission.ipynb"])
    args = parser.parse_args()
    for name in args.notebooks:
        execute_notebook(ROOT / "notebooks" / name)


if __name__ == "__main__":
    main()
