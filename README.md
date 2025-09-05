# MSc Groundwater GWL Project

This repository contains the code only for my project on groundwater level modelling in UK catchments using a GAT-LSTM architecture.


## CEDA Credentials

To access HadUK-Grid data via OPeNDAP you will need to add your own credentials to the project:

1. Create a file called `ceda_credentials.netrc` in the project root with this format:

    machine dap.ceda.ac.uk
    login your_username
    password your_password

2. Run:

```bash
chmod 600 ceda_credentials.netrc
