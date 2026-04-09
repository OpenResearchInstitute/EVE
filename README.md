# EVE
Earth Venus Earth Project work

## An Earth-Venus-Earth Link Budget from Open Research Institute
This repository contains a detailed link budget analysis for Earth-Venus-Earth (EVE) communications. It begins with a Python dataclass for each fixed earth station. Dataclasses are a type of Python object where only variables are declared. No methods are included. The site-specific dataclasses have parameters that are true for the site regardless of the target. Following the site-specific dataclasses are link budget classes, which contain target-specific values and methods that return various gains and losses. The most important output of a link budget object is a carrier to noise ratio at a particular receive bandwidth. A link budget class inherits a particular site dataclass. The site-specific dataclasses and the link budgets can be mixed and matched. This gives flexibility, as a link budget for a particular target, such as EVE, can be calculated for different sites by having that link budget inherit different site-specific dataclasses.

## Documents

- **Link_Budget_Modeling.ipynb** the Jupyter lab notebook that does the link analysis
- **venus_albedo_mapper.py** helper classes for the dynamic albedo calculation
- **test_venus_albedo_mapper.py** test script to prove out venus_albedo_mapper.py
- **eve-signal-processing-diagram.svg** embedded image illustrating concepts for the Zadoff Chu section
- **eve_cnr_measusrement.ipynb** validation of the ORI Link Budget using March 2025 CAMRAS data
- **Validation of the ORI Earth-Venus-Earth Link Budget.pdf** white paper about the validation of the ORI Link Budget using March 2025 CAMRAS data

## How to Get This Running

Clone this repository, set up the python environment of your choice, and install dependencies. Run jupyter lab in the EVE directory, and you should be able to open Link_Budget_Modeling.ipynb. Run all cells. Bring any problems you can't solve with getting it up and running to ORI. See https://openresearch.institute/getting-started for how to get in touch for assistance. Issues and PRs are welcome. 
