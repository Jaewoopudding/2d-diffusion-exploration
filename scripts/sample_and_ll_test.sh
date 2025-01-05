distributions=("swissroll" "8gaussians" "moons" "rings" "checkerboard" "2spirals" "uniform" "elliptic_paraboloid")

# Initialize CUDA device index
cuda_devices=(5 4)
device_index=0

# Iterate over each distribution
for distribution in "${distributions[@]}"
do
    # Select CUDA device in a round-robin fashion
    CUDA_VISIBLE_DEVICES=${cuda_devices[device_index]} python likelihood.py --distribution "$distribution" &

    # Update device index to toggle between 0 and 1
    device_index=$((1 - device_index))
done

# Wait for all background processes to complete
wait