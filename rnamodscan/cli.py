"""Project entry point; pipeline implementations live in their own packages."""
import argparse
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'predict':
        from prediction.cli import main as prediction_main
        return prediction_main(arguments[1:])
    if arguments and arguments[0] == 'train':
        from training.cli import main as training_main
        return training_main(arguments[1:])
    if arguments and arguments[0] == 'quality':
        from utils.quality.cli import main as quality_main
        return quality_main(arguments[1:])
    if arguments and arguments[0] == 'references':
        from utils.references.cli import main as references_main
        return references_main(arguments[1:])
    parser = argparse.ArgumentParser(
        description='rnaModScan: train models, predict sites and prepare reference catalogs',
        epilog=('Start a new training run: rnamodscan train init --help. '
                'Check inputs: rnamodscan train check --help. '
                'Import positive sites: rnamodscan references build --help.'))
    parser.add_argument('command', choices=('train', 'predict', 'references', 'quality'))
    parser.parse_args(arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
