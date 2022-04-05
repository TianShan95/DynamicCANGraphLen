import argparse

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('-c', action='store_true', default=False)

print(parser.parse_args())
