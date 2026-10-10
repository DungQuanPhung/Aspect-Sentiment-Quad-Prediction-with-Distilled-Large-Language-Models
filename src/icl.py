import os
from models import execute_prompt
import random
import ast
from collections import Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from collections import defaultdict
import copy
import numpy as np

examples, icl_template = list(), list()

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def extract_examples(text):
    """Extracts examples from a given text."""
    global examples
    for line in text.split('\n'):
        if '####' in line:
            parts = line.split('####')
            quadruples = ast.literal_eval(parts[1])
            quadruples = [quad[:4] for quad in quadruples]
            examples.append((parts[0], repr(quadruples)))

def fill_examples(dataset_name):
    """Fills the global examples list from a dataset."""
    global examples
    data_path = os.path.join(os.getcwd(), "data", "raw")
    input_file = os.path.join(data_path, dataset_name, 'train.txt')
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    extract_examples(text)

def get_example_indices(*, dataset_name, number_of_examples=2, random_seed=32):
    """Gets a stratified sample of example indices from a dataset."""
    global examples
    fill_examples(dataset_name)
    random.Random(random_seed).shuffle(examples)
    classes_index = list()
    for example in examples:
        lit = ast.literal_eval(example[1])
        cat_pol = set()
        if len(lit) == 0:
            cat_pol.add(frozenset({}))
        else:
            for l in lit:
                cat_pol.add(frozenset({l[1], l[2]}))
        classes_index.append(cat_pol)

    unique_labels = defaultdict(int)
    for subset in classes_index:
        if len(subset) == 0:
            unique_labels[frozenset({})] += 1
            continue
        for s in subset:
            unique_labels[s] += 1

    unique_categories = set()
    for froz in unique_labels.keys():
        s_froz = set(froz)
        s_froz.discard("positive")
        s_froz.discard("negative")
        s_froz.discard("neutral")
        try:
            unique_categories.add(next(iter(s_froz)))
        except StopIteration:
            unique_categories.add("")
    categories = copy.copy(unique_categories)
    if "" in categories:
        categories.remove("")

    example_indices = list()
    if number_of_examples == "all":
        example_indices = list(range(len(classes_index)))
    elif number_of_examples <= len(unique_categories):
        for idx, _class in enumerate(classes_index):
            if len(example_indices) == number_of_examples:
                break
            for cl in _class:
                s_cl = set(cl)
                s_cl.discard("positive")
                s_cl.discard("negative")
                s_cl.discard("neutral")
                try:
                    s_cl = next(iter(s_cl))
                except StopIteration:
                    s_cl = ""
                break
            if s_cl in unique_categories:
                example_indices.append(idx)
                unique_categories.remove(s_cl)
    else:
        sorted_unique_labels = sorted(unique_labels.items(), key=lambda x: x[1], reverse=True)
        label_counts = defaultdict(int)
        remaining = copy.copy(number_of_examples)
        try:
            while True:
                for s_id, tup in enumerate(sorted_unique_labels):
                    if tup[1] <= 0:
                        break
                    label_counts[tup[0]] += 1
                    sorted_unique_labels[s_id] = (tup[0], tup[1] - 1)
                    remaining -= 1
                    if remaining <= 0:
                        raise StopIteration
        except StopIteration:
            pass
        print("Gini index:", gini(np.array(list(label_counts.values()))))
        example_indices = list()
        for idx, _class in enumerate(classes_index):
            for cl in _class:
                if label_counts[cl] > 0:
                    label_counts[cl] -= 1
                    example_indices.append(idx)
        example_indices = list(set(example_indices))
        if len(example_indices) < number_of_examples:
            remaining = number_of_examples - len(example_indices)
            for idx, _class in enumerate(classes_index):
                if idx not in example_indices:
                    example_indices.append(idx)
                    remaining -= 1
                    if remaining <= 0:
                        break
    return example_indices, categories

def fill_icl_template_for_icl(dataset_name, number_of_examples=2, random_seed=32):
    """Fills the in-context learning template for the ICL task."""
    global icl_template, examples
    examples = list()
    example_indices, categories = get_example_indices(dataset_name=dataset_name, number_of_examples=number_of_examples,
                                                      random_seed=random_seed)
    icl_template = [
        {"role": "user", "content":
            f"""Parse the following text review in an Aspect Sentiment Quadruple Prediction format.
            The aspects and opinions must be terms existing in the input text or 'NULL' if non-existing.
            The category type is one in the predefined list: {categories}.
            The sentiment is 'positive', 'negative' or 'neutral'.
            Do not ask me for more information, as I am unable to provide it; just try your best to finish the task.
            The quadruples have the format [['<aspect>', '<category>', '<polarity>', '<opinion>'], [...], ...].
            Please parse the text below."""},
        {"role": "assistant", "content":
            "Please provide the text review you want me to parse into Aspect Sentiment Quadruple Prediction format."}]
    for idx, example in enumerate(examples):
        if idx in example_indices:
            icl_template.append({"role": "user", "content": example[0]})
            icl_template.append({"role": "assistant", "content": example[1]})
    return icl_template

def fill_icl_template_for_cot_agents(dataset_name, number_of_examples=2, random_seed=32):
    """Fills the in-context learning template for the CoT agents task."""
    global icl_template, examples
    examples = list()
    example_indices, categories = get_example_indices(dataset_name=dataset_name, number_of_examples=number_of_examples,
                                                      random_seed=random_seed)
    print("len(examples)")
    print(len(examples))
    icl_template = [
        {"role": "user", "content": """Now, parse the following review text in an Aspect Sentiment Quadruple Prediction format.
         The quadruples have the format [['<aspect>', '<category>', '<polarity>', '<opinion>'], [...], ...].  
         Please parse the text below."""},
        {"role": "assistant", "content":
            "Please provide the text review you want me to parse into Aspect Sentiment Quadruple Prediction format."}]
    for idx, example in enumerate(examples):
        if idx in example_indices:
            icl_template.append({"role": "user", "content": example[0]})
            icl_template.append({"role": "assistant", "content": example[1]})

    icl_template.append(
        {"role": "user", "content": """This is great! Perform the same type of Aspect Sentiment Quadruple Prediction parsing for the aspects, categories, and sentiments you extracted earlier regarding the first review text."""})

def run_icl(history, model="gpt-4-turbo"):
    """Runs the in-context learning task."""
    history.extend(icl_template)
    out = execute_prompt(chat_history=history[1:-1], prompt=history[-1]['content'],
                         system_instruction=history[0]['content'], model=model)
    return out, history[1:]