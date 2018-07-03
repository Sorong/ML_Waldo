import argparse

from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>",
                        help='Spezifiert den Modus: \'train\' oder \'evaluate\'')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/imageset",
                        help='Directory of the Waldo Dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    args = parser.parse_args()

    if args.command == "train":
        t = Trainer()
        t.train()

