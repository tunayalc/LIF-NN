# LIF Neural Network (Spiking)

This project implements a Leaky Integrate-and-Fire (LIF) spiking neural network
with STDP learning and a Boltzmann-inspired update for hidden states.

## What It Does

- Converts input data into spike trains
- Trains using STDP with homeostatic adjustments
- Produces class predictions and a submission CSV

## Data

Expected files in the project root (not included):

- train.csv (must include a "label" column)
- test.csv

If the files are missing, the script generates a small synthetic dataset
so you can run the pipeline end-to-end.

## Run

python LIF_MODEL.py

The script writes `submission_snn.csv` to the project root.

## Requirements

Install dependencies with:

pip install -r requirements.txt

## Notes

This is a research/learning implementation and is not optimized for speed.