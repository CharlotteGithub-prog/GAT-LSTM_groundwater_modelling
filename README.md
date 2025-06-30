# MSc Groundwater GWL Project

This repository contains the code and data for my MSc dissertation on groundwater level modelling in UK catchments using GNN-based architectures.


## CEDA Credentials

To access HadUK-Grid data via OPeNDAP:

1. Create a file called `ceda_credentials.netrc` in the project root with this format:

    machine dap.ceda.ac.uk
    login your_username
    password your_password

2. Run:

```bash
chmod 600 ceda_credentials.netrc