#!/bin/bash
# Wait for training to complete
PID=3091275

echo "Monitoring training process (PID: $PID)..."

while kill -0 $PID 2>/dev/null; do
    sleep 30
done

echo "Training complete!"
echo "==========================================="
tail -100 /home/joao/Documentos/code/softtor/molting/experiments/fine-tuning/full_training.log
