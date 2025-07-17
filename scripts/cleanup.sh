#!/bin/bash

# Customer Segmentation Modern - Cleanup Script
# This script removes all resources created by the solution accelerator

set -e

echo "🧹 Cleaning up Customer Segmentation Solution Accelerator..."
echo "========================================================="

# Get target environment (default to dev)
TARGET=${1:-dev}
echo "📋 Cleaning up target: $TARGET"

# Warning prompt
echo "⚠️  WARNING: This will delete all resources including:"
echo "  • Delta Live Tables pipeline"
echo "  • Unity Catalog tables and data"
echo "  • Databricks jobs"
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

# Destroy the bundle
echo "🗑️  Destroying bundle resources..."
databricks bundle destroy --target $TARGET --auto-approve

echo "✅ Cleanup completed successfully!"
echo ""
echo "💡 Note: This script removes the Databricks resources but preserves:"
echo "  • Your local code and configuration files"
echo "  • Git repository and branches"