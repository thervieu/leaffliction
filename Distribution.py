import os
import sys
import matplotlib.pyplot as plt
from typing import TypedDict


class CountDict(TypedDict):
    directory_name: str
    number_of_elements: int


def help() -> None:
    print("help:\n\tDescription.py [Directory]")


def getCountDictionary(directory: str) -> CountDict:
    # dictionnary with subdir as keys and
    # number of photos in each subdir as values
    count_dict: CountDict = CountDict()
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,subdir)) is False:
            return print("{} is not a directory".format(os.path.join(directory,subdir)))
        count_dict[subdir] = len(os.listdir(os.path.join(directory,subdir)))
    return count_dict

def getNameValue(leaf:str) :
    count_dict = getCountDictionary(leaf)
    names = list(count_dict.keys())
    values = list(count_dict.values())
    plt.bar(range(len(count_dict)), values, tick_label=names, color=['blue', 'orange', 'green', 'red'])
    plt.figure()
    plt.pie(values, labels=names)
    plt.show()
    

def main() -> None:
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))
    leafName = os.path.join(sys.argv[1], '')
    print(leafName)
    getNameValue(leafName)

if __name__ == "__main__":
    main()
