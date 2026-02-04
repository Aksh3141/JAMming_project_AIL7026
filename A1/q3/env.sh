#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python dependencies..."
pip install --quiet --break-system-packages numpy networkx 2>/dev/null || \
pip install --quiet numpy networkx

echo "Dependencies installed (numpy, networkx)"

GASTON_BIN="$SCRIPT_DIR/gaston"

if [ ! -x "$GASTON_BIN" ]; then
    echo "Gaston binary not found – attempting to compile..."

    if [ -d "$SCRIPT_DIR/gaston-1.1" ]; then
        cd "$SCRIPT_DIR/gaston-1.1"
        
        if ! grep -q '#include <getopt.h>' main.cpp 2>/dev/null; then
            echo "Adding missing header to main.cpp..."
            sed -i '1i #include <getopt.h>' main.cpp
        fi
        
        echo "Compiling Gaston..."
        make clean 2>/dev/null || true
        make -j$(nproc) 2>&1 | tail -5  
        
        if [ -f "gaston" ]; then
            cp gaston "$GASTON_BIN"
            chmod +x "$GASTON_BIN"
            echo "Gaston compiled successfully"
        else
            echo "ERROR: Compilation succeeded but binary not found"
        fi
        
        cd "$SCRIPT_DIR"
    else
        echo "WARNING: gaston-1.1 directory not found at $SCRIPT_DIR/gaston-1.1"
    fi

    if [ ! -x "$GASTON_BIN" ]; then
        echo "ERROR: Could not compile gaston automatically."
        echo "  Place the compiled 'gaston' binary at: $GASTON_BIN"
        echo "  And run: chmod +x $GASTON_BIN"
        exit 1
    fi
else
    echo "Gaston binary already exists"
fi

if [ -x "$GASTON_BIN" ]; then
    echo "Testing Gaston binary..."
    if "$GASTON_BIN" 2>&1 | grep -q "gaston" || [ $? -eq 1 ]; then
        echo "Gaston binary is executable"
    else
        echo "WARNING: Gaston binary may not be working correctly"
    fi
fi

# Make other scripts executable
chmod +x "$SCRIPT_DIR/identify.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/convert.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/generate_candidates.sh" 2>/dev/null || true

echo ""
echo "========================================="
echo "env.sh complete"
echo "========================================="
echo "Python: $(python3 --version)"
echo "NumPy: $(python3 -c 'import numpy; print(numpy.__version__)')"
echo "NetworkX: $(python3 -c 'import networkx; print(networkx.__version__)')"
echo "Gaston binary: $GASTON_BIN"
echo "========================================="