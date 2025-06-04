#!/bin/bash
# filepath: /home/siavashi/mohamamd/tests/moe-serving/launch_example.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO]${NC} Starting offline_llama_8b example with optimized settings..."

# Store original settings for restoration
declare -A ORIGINAL_CPU_STATUS
ORIGINAL_BOOST_STATUS=""

# Function to restore original settings
cleanup() {
    echo -e "\n${YELLOW}[CLEANUP]${NC} Restoring original system settings..."
    
    # Restore hyperthreading
    if [ ${#ORIGINAL_CPU_STATUS[@]} -gt 0 ]; then
        echo -e "${YELLOW}[CLEANUP]${NC} Restoring hyperthreading..."
        for cpu_num in "${!ORIGINAL_CPU_STATUS[@]}"; do
            cpu_file="/sys/devices/system/cpu/cpu${cpu_num}/online"
            if [ -f "$cpu_file" ]; then
                echo "${ORIGINAL_CPU_STATUS[$cpu_num]}" | sudo tee "$cpu_file" > /dev/null 2>&1 || true
            fi
        done
    fi
    
    # Restore frequency boost
    if [ ! -z "$ORIGINAL_BOOST_STATUS" ]; then
        echo -e "${YELLOW}[CLEANUP]${NC} Restoring frequency boost..."
        echo "$ORIGINAL_BOOST_STATUS" | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}[CLEANUP]${NC} System settings restored successfully"
}

# Set up signal handlers for cleanup
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM

# Check if running as root for system modifications
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} Please run this script as a regular user (it will use sudo when needed)"
    exit 1
fi

# Check if sudo is available
if ! command -v sudo &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} sudo is required but not installed"
    exit 1
fi

# Store current hyperthreading status for each CPU
echo -e "${YELLOW}[CONFIG]${NC} Storing current CPU states..."
for cpu in /sys/devices/system/cpu/cpu*/online; do
    if [ -f "$cpu" ]; then
        cpu_num=$(basename $(dirname $cpu) | sed 's/cpu//')
        if [ $((cpu_num % 2)) -eq 1 ]; then
            ORIGINAL_CPU_STATUS[$cpu_num]=$(cat "$cpu" 2>/dev/null || echo "1")
        fi
    fi
done

# Store current frequency boost status
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    ORIGINAL_BOOST_STATUS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "1")
fi

echo -e "${YELLOW}[CONFIG]${NC} Temporarily disabling hyperthreading..."
# Disable hyperthreading (disable odd-numbered CPUs)
for cpu in /sys/devices/system/cpu/cpu*/online; do
    if [ -f "$cpu" ]; then
        cpu_num=$(basename $(dirname $cpu) | sed 's/cpu//')
        if [ $((cpu_num % 2)) -eq 1 ]; then
            echo 0 | sudo tee "$cpu" > /dev/null 2>&1 || echo -e "${YELLOW}[WARNING]${NC} Could not disable CPU $cpu_num"
        fi
    fi
done

echo -e "${YELLOW}[CONFIG]${NC} Temporarily disabling frequency boost..."
# Disable frequency boost
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null 2>&1 || echo -e "${YELLOW}[WARNING]${NC} Could not disable frequency boost"
fi

# Check if numactl is available
if ! command -v numactl &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} numactl is required but not installed. Please install it with: sudo apt-get install numactl"
    exit 1
fi

# Check if NUMA node 0 exists
if ! numactl --hardware | grep -q "node 0"; then
    echo -e "${RED}[ERROR]${NC} NUMA node 0 not found"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} System configured successfully"
echo -e "${GREEN}[INFO]${NC} Starting Python script with CUDA_VISIBLE_DEVICES=2, pinned to NUMA node 0..."
echo -e "${GREEN}[INFO]${NC} Output will be saved to res.out"

# Change to the script directory
cd "$(dirname "$0")"

# Launch the Python script with specified settings
# - CUDA_VISIBLE_DEVICES=2: Use GPU 2
# - numactl --cpunodebind=0 --membind=0: Pin to NUMA node 0
# - tee: Pipe output to both terminal and file in real time
# - pipefail: Ensure we catch errors from the Python script
set -o pipefail
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 python ./examples/offline_llama_8b.py 2>&1 | tee res.out

echo -e "${GREEN}[SUCCESS]${NC} Script completed successfully"
echo -e "${GREEN}[INFO]${NC} Output saved to res.out"