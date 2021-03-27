# Creating a kernel
1. kaggle kernels init -p working/
2. edit kernel.json
NB:
- A Notebook slug (id) is always the title lowercased with dashes (-) replacing spaces and removing special characters. (slug: end part of the URL, no underscore in URLs).
- /!\ code file should avoid underscore / dashes if they are utility scripts

- To add a dataset:
"dataset_sources": [
    "pseudo/dataset-name"
]

- To add a utility script:
"kernel_sources": [
    "pseudo/utility_script_name"
]

3. kaggle kernels push -p /path/to/kernel

# Pulling
1. kaggle kernels pull pseudo/kernel-name -p working/
(-m : to get the metadata file)

# Retrieve kernel's output
1. kaggle kernels output pseudo/kernel-name -p output/ 

