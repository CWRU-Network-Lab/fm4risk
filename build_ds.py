
# training dataset builder -- nathaniel hahn
from datasets import load_dataset


def filter_dataset(split):
        # Load the CodeSearchNet dataset
        codesearchnet = load_dataset('code_search_net', "python")

        # Get the training split
        train_dataset = codesearchnet['train']

        # sort by keyword for indepth testing on specific functions
        if split == "sort":
            sort_examples_dataset = train_dataset.filter(lambda example: "sort" in example["func_code_string"].lower() or "sort" in example["func_documentation_string"].lower())
        else:
            sort_examples_dataset = train_dataset
        # Print the number of examples after filtering
        print("Number of examples :", len(sort_examples_dataset)) 
        return sort_examples_dataset
