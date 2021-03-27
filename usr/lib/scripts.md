# Creating a script
Scripts should be in a folder of the same name.
/!\ After running the script, Kaggle will name it (code_file) according to the title, with underscores replacing dashes.
→ it is thus easier to choose a same name for all fields (id, title, code_file) using just letters, without space, underscore or dash.

Created like kernels with a specific parameters:
"kernel_type": "script",
"keywords": [
    "utility script"
]

## Known issues
Sometimes, after pushing a script (for the first time?) and running it on Kaggle, it won't appear in the output of the run. Modifying a line of the script and pushing it again solves this.