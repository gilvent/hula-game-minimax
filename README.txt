# [NYCU Project] Custom Hula Board Game - AI Implementation using Minimax algorithm

## How to run
### Prerequisites
* Python 3 (The version used in development `Python 3.12.7`)
* pip (package manager generally built-in with Python installation)


### Running the project
1. Install pygame using `python -m pip install -r requirements.txt` (Recommended) or run `pip install pygame`
2. Run the program using `python ./src/hula_version2.py <player 1> <player 2>`

Examples:
* `python ./src/hula_version2.py random AI`
* `python ./src/hula_version2.py human AI`
* `python ./src/hula_version2.py AI AI`


## Implementation notes
* The AI implementation also uses pygame timer for stopping criterion. The algorithm will stop after 28 seconds.