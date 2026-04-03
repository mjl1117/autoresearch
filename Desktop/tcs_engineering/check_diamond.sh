#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
echo "=== tcs-env diamond ==="
ls tcs-env/bin/diamond 2>/dev/null || echo "NOT in tcs-env/bin"
echo "=== homebrew diamond ==="
ls /opt/homebrew/bin/diamond 2>/dev/null || echo "NOT in /opt/homebrew/bin"
echo "=== which diamond ==="
which diamond 2>/dev/null || echo "diamond not on PATH"
echo "=== diamond version ==="
tcs-env/bin/diamond version 2>/dev/null || /opt/homebrew/bin/diamond version 2>/dev/null || echo "diamond not runnable"
