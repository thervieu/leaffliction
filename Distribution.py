import os
import sys
import matplotlib.pyplot as plt
from typing import TypedDict


class CountDict(TypedDict):
    directory_name: str
    number_of_elements: int


def help() -> None:
    print("help:\n\tDescription.py [Directory]")


def main() -> None:
    # argument
    if len(sys.argv) != 2:
        return print("")
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))
    # dictionnary with subdir as keys and
    # number of photos in each subdir as values
    count_dict: CountDict = CountDict()
    for subdir in os.listdir(sys.argv[1]):
        if os.path.isdir(sys.argv[1]+subdir) is False:
            return print("{} is not a directory".format(sys.argv[1]+subdir))
        count_dict[subdir] = len(os.listdir(sys.argv[1]+subdir))
    names = list(count_dict.keys())
    values = list(count_dict.values())

    plt.bar(range(len(count_dict)), values, tick_label=names)
    plt.figure()
    plt.pie(values, labels=names)
    plt.show()


if __name__ == "__main__":
    main()
