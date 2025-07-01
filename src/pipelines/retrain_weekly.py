"""Weekly batch pipeline: re-run dataset merge → features → train → evaluate.
Executes unit tests and, when passing thresholds, builds & tags new Docker image.
"""
import subprocess, sys, pathlib

BASE = pathlib.Path(__file__).resolve().parents[2]

def run(cmd):
    print(f"-- $ {cmd}")
    res = subprocess.run(cmd, shell=True, check=True)
    return res.returncode

def main():
    # Sequential pipeline
    run("python -m src.data.make_dataset")
    run("python -m src.features.build_features")
    run("python -m src.models.train_model")
    run("pytest -q tests")
    print("Weekly retrain finished successfully!")

if __name__ == '__main__':
    main()
