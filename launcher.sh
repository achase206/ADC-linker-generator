#!/bin/bash

echo "=========================================="
echo "      ADC Linker LSTM Generator"
echo "=========================================="
echo "1. Train Models (LSTM_train.py)"
echo "2. Generate/Analyze (LSTM_run.py)"
echo "3. Exit"
echo "=========================================="

read -p "Select an option [1-3]: " option

case $option in
    1)
        echo "Launching Training Script..."
        python LSTM_train.py
        ;;
    2)
        echo "Launching Run Script..."
        python LSTM_run.py
        ;;
    3)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option selected."
        exit 1
        ;;
esac