import argparse
from run_cartpole import train_cartpole
from run_mountaincar import train_mountaincar
from run_acrobot import train_acrobot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole",
                        choices=["cartpole", "mountaincar", "acrobot"])
    args = parser.parse_args()

    if args.env == "cartpole":
        train_cartpole()
    elif args.env == "mountaincar":
        train_mountaincar()
    elif args.env == "acrobot":
        train_acrobot()
