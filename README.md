# Network: An Opinion Space Agent-Based Model

Model of filter bubbles and opinion dynamics.

Created by Scott Vella <scott.vella@anu.edu.au> for the Next Threats project,
National Security College <https://nsc.anu.edu.au/>, Australian National University <https://anu.edu.au/>.

Main contact: Professor Roger Bradbury <roger.bradbury@anu.edu.au>

## Simple Usage

Clone the repository, and optionally setup a virtual environment for the project.

Install the dependencies

`pip install -r requirements.txt`

For some cool animations, and to replicate the summary graphs presented in _the paper_ run:

`python network/main.py`

### ModuleNotFoundError or ImportError: No module named _tkinter

To run the animations and to view the graphs requires Matplotlib, which in turn requires tkinter. tkinter can be finicky to install and link in virtualenvs - especially when managing multiple versions of python, and on MacOS. Since the instructions vary significantly from system to system, please search for the most up-to-date instructions on how to install and link tkinter for your system.

## Running your own experiments

```
from options import Arguments
from main import animate, run_simulation

animate(Arguments(...)) # For visualisation
# or
run_simulation(Arguments(...)) # For headless
```

The available arguments are extensively documented in `network/options.py`.

## Extending the models

In case the dynamics you'd like to model are not yet included as standard here, the model is easy to extend.

We recommend placing new parameters as keyword arguments of the options.Arguments class, then adding the relevant functionality to model.Agent and model.Context as required.

Easy extensions include adding distribution types to each of the various components (which follow the common existing format).

More complex extensions would include localised (e.g. multi-variate gaussian) opinions, major and minor party politics, and message-passing on the network.

## Acknowledgements

This codebase has evolved from a previous iteration developed by Dmitry Brizhinev <dmitry.brizhinev@anu.edu.au> (also for the Australian National University National Security College). It is available at <https://github.com/ANU-NSC-Strategy-Statecraft-Cyberspace/network.git>. However, it is no longer maintained and whilst it requires Python 3, it may not run properly on modern versions of python (i.e. anything above 3.5).

## How to cite

Cite as

_TBC_

or with bibtex

_TBC_
