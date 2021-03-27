# %% 
import sys, os

# Add utility_scripts in the current path so that they can be imported directly just like in interactive mode
sys.path.append(os.path.abspath("../usr/lib/"))
for script_folder in os.listdir("../usr/lib/"):
    sys.path.append(os.path.abspath("../usr/lib/"+script_folder))
# %%
# Conditional construct depending on where the kernel is run.
loc = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')
if loc == 'Interactive' or loc == 'Localhost':
    conf = {
        'train_size': 2000,
        'test_size': 2000,
        'epochs': 2,
    }
# When it is run after an api push.
elif loc == 'Batch':
    conf = {
        'train_size': -1,
        'test_size': -1,
        'epochs': 5
    }
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'