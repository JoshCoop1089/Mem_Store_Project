# Mem_Store_Project

### Dependencies and env creation
```
conda env create -f environment.yml
```

Python 3.9 only required for the LOWESS smoothing on results graphs in contextual_choice_sl.py

Run experiment_run.py to do tests on multiple memory types for the same set of experiental setups

Change the graph saving location in experiment_run.py if needed

Tensorboard logging of results, loss, accuracy disabled by default.

Change exp_settings['tensorboard_logging'] boolean in run_experiment function in contextual_choice_sl.py if needed

### File Structure:
```
├── README.md
├── environment.yml
├── figs                                # Save location for graphs created
└── src
    ├── experiment_run.py               # quick driver for running multiple experiments from contextual_choice_sl.py
    ├── contextual_choice_sl.py         # train the model on a contextual choice task, in .py
    ├── model   
    │   ├── A2C.py                      # an advantage actor critic agent
    │   ├── DND.py                      # the memory module 
    │   ├── embedding_model.py          # the embedder which goes between hidden states and memory keys
    │   ├── DNDLSTM.py                  # a LSTM-based A2C agent with DND memory 
    │   ├── utils.py
    └── └── __init__.py
    ├── task
    │   ├── ContextBandits.py           # the definition of the contextual choice task
    │   └── __init__.py
    └── utils.py
```
