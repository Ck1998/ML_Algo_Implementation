from modules.utils.constants import DATASET_MAPPINGS
from modules.models import ALGO_MAPPING
from os import system, name


def clear():
    if name == 'nt':
        system('cls')
    else:
        system('clear')


if __name__ == "__main__":
    cont = True
    while cont:
        clear()
        print("Menu")
        # Choosing a an algo type
        print("Choose the type of model you want to run: ")
        for i in ALGO_MAPPING.keys():
            print(f"{i}. {ALGO_MAPPING[i]['name']}")
        print("Enter '0' to restart.")
        algo_type = int(input("Your choice > "))
        if algo_type == 0:
            continue

        # Choosing an algorithm
        print("\nSelect an Algorithm -")
        for i, val in ALGO_MAPPING[algo_type]["algorithms"].items():
            print(f"{i}. {ALGO_MAPPING[algo_type]['algorithms'][i]['name'].replace('_', ' ')}")
        print("Enter '0' to restart.")
        algo_id = int(input("Your choice > "))
        if algo_id == 0:
            continue
        algo_obj = ALGO_MAPPING[algo_type]['algorithms'][algo_id]['obj']

        if algo_type == 3:
            algo_obj().run()
        else:
            datasets_to_use = ALGO_MAPPING[algo_type]["datasets"]

            # Choosing a dataset
            print("\nSelect a dataset to use -")
            for i, val in DATASET_MAPPINGS.items():
                if i in datasets_to_use:
                    print(f"{i}. {DATASET_MAPPINGS[i]['name'].replace('_', ' ')}")
            print("Enter '0' to restart.")
            d_id = int(input("Your choice > "))
            if d_id == 0:
                continue
            print("Running algorithm on the selected dataset")
            algo_obj(dataset_id=d_id, prefix=False).run()

            if input("Do you wish to continue? [Y/N]: ").lower() != "y":
                cont = False