#!/bin/bash

# Customer Segmentation Modern - Deployment Script
# This script deploys the Customer Segmentation solution accelerator using Databricks Asset Bundles

set -e

echo "🚀 Deploying Customer Segmentation Solution Accelerator..."
echo "================================================"

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI is not installed. Please install it first:"
    echo "   pip install databricks-cli"
    exit 1
fi

# Check if user is authenticated
if ! databricks current-user me &> /dev/null; then
    echo "❌ Not authenticated with Databricks. Please run:"
    echo "   databricks configure"
    exit 1
fi

echo "✅ Databricks CLI is installed and authenticated"

# Validate databricks.yml exists
if [ ! -f "databricks.yml" ]; then
    echo "❌ databricks.yml not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Found databricks.yml configuration"

# Get target environment (default to dev)
TARGET=${1:-dev}
echo "📋 Deploying to target: $TARGET"

# Deploy the bundle
echo "⚙️  Validating and deploying bundle..."
databricks bundle validate --target $TARGET

echo "🔧 Deploying resources..."
databricks bundle deploy --target $TARGET

echo "✅ Deployment completed successfully!"
echo ""
echo "🎯 Next Steps:"
echo "1. Navigate to your Databricks workspace"
echo "2. Find the 'Customer Segmentation Pipeline' in Delta Live Tables"
echo "3. Run the pipeline to generate synthetic data and perform segmentation"
echo "4. Open the Business Insights notebook for interactive visualizations"
echo ""
echo "📊 Pipeline Components:"
echo "  • Data Generation: Creates 10K customers, 250K transactions"
echo "  • Segmentation Analysis: RFM + behavioral clustering"  
echo "  • Business Insights: Interactive Plotly dashboards"
echo ""
echo "💡 Pro tip: The solution uses Unity Catalog managed tables for easy governance!"