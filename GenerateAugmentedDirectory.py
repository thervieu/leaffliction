import os, sys, random
from datetime import datetime

from Augmentation import augment
from Distribution import getCountDictionary


def help() -> None:
    print("help:\n\GenerateAugmentedDirectory.py [directory]")


def main():
    random.seed(datetime.now().timestamp())
    if len(sys.argv) != 2:
        return help()
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))
    # delete max from dict
    count_dict = getCountDictionary(sys.argv[1])
    max_key = max(count_dict, key=lambda key: count_dict[key])
    max_count = count_dict[max_key]
    count_dict.pop(max_key)
    # augmentation each subdirectory
    for key in count_dict:
        print('Augmenting', key)
        augment_number = int((max_count - count_dict[key]) / 7)
        print('Augmenting {} images'.format(augment_number))
        images = os.listdir(os.path.join(sys.argv[1], key))
        for _ in range(augment_number):
            cur_img = random.choice(images)
            augment(os.path.join(sys.argv[1], key,cur_img))
            images.remove(cur_img)
        print('Done\n')
    print('All subdirectories augmented')


if __name__ == "__main__":
    main()