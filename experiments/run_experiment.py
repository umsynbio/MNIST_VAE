"""Entrypoint wrapper so you can call with -m experiments.run_experiment"""
import argparse
from training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()