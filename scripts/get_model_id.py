import argparse
import yaml

parser = argparse.ArgumentParser(description="Get model id")
parser.add_argument(
    "--config",
    default="experiments/configs/global.yaml",
    type=str,
    metavar="CONFIG",
    help="Global configuration file",
)
parser.add_argument(
    "--name",
    type=str,
    metavar="NAME",
    help="The name to search for",
)


def get_id(name, models, key="name"):
    return tuple(map(lambda model: model[key], models)).index(name)


def main():
    args = parser.parse_args()
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)
    id = get_id(
        args.name,
        cfg["models"],
    )
    print(id)


if __name__ == "__main__":
    main()
