# Customer Segmentation Solution Accelerator

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

**Transform customer data into actionable business insights with modern RFM analysis and behavioral segmentation.**

## 🎯 Industry Use Case

Customer segmentation is the foundation of modern marketing strategy, enabling businesses to:
- **Increase revenue by 20%** through targeted campaigns
- **Improve customer lifetime value** across all segments  
- **Enhance marketing efficiency** with precision targeting
- **Reduce customer churn** through proactive engagement

This solution demonstrates a paradigm shift from traditional demographic-only segmentation to **behavioral-driven customer intelligence**, using RFM (Recency, Frequency, Monetary) analysis combined with advanced clustering techniques to create actionable customer segments.

## 🚀 What is Customer Segmentation?

Customer segmentation divides your customer base into distinct groups based on shared characteristics and behaviors. This solution creates **6 distinct customer segments**:

1. **High-Value Loyalists** - Premium customers generating highest revenue
2. **Frequent Shoppers** - Regular customers with consistent purchase patterns
3. **Discount Hunters** - Price-sensitive customers responding to promotions
4. **Occasional Buyers** - Sporadic purchasers needing engagement
5. **New/Inactive Customers** - Recent sign-ups or dormant accounts
6. **Category Specialists** - Customers focused on specific product categories

Each segment receives tailored strategies with **150-200% expected ROI** for high-value segments.

## 📦 Installation

This solution uses [Databricks Asset Bundle](https://docs.databricks.com/en/dev-tools/bundles/index.html) for deployment:

```bash
# Clone the repository
git clone https://github.com/databricks-industry-solutions/customer-segmentation.git
cd customer-segmentation

# Deploy to Databricks
databricks bundle deploy

# Run the complete workflow
databricks bundle run customer_segmentation_demo_install
```

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- Databricks CLI installed and configured
- Cluster creation permissions

## 🏗️ Project Structure

```
customer-segmentation/
├── databricks.yml                 # Databricks Asset Bundle configuration
├── notebooks/
│   ├── 01_Data_Setup.py          # Synthetic data generation
│   ├── 02_Segmentation_DLT.py    # DLT pipeline for segmentation
│   └── 03_Business_Insights.py   # Business visualizations
└── .github/workflows/             # CI/CD automation
```

## 🔄 Segmentation Pipeline

The solution implements a **3-stage customer segmentation pipeline**:

### Stage 1: Data Setup
- Generates **1,000 synthetic customers** with realistic demographics
- Creates **transaction history** with seasonal patterns and behavioral variety
- Stores data in **Unity Catalog managed tables**

### Stage 2: Segmentation Analysis (DLT)
- **RFM Analysis**: Calculates Recency, Frequency, and Monetary scores
- **Behavioral Clustering**: Groups customers by purchase patterns
- **Segment Profiles**: Creates business-ready segment characteristics

### Stage 3: Business Insights
- **Interactive Visualizations**: 5 essential charts using Plotly
- **Actionable Recommendations**: ROI-focused strategies per segment
- **Executive Summary**: Business-ready insights and next steps

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```yaml
# databricks.yml variables
variables:
  catalog_name: your_catalog_name
  schema_name: your_schema_name
```

## 📊 Expected Business Impact

Based on industry benchmarks, implementing this segmentation strategy delivers:
- **20% average revenue lift** through targeted campaigns
- **15-30% improvement** in customer lifetime value
- **40% increase** in marketing campaign effectiveness
- **25% reduction** in customer acquisition costs

## 🎨 Visualization Highlights

The solution includes 5 essential visualizations:
1. **Customer Distribution** - Segment size analysis
2. **Revenue Distribution** - Revenue concentration by segment
3. **Performance Metrics** - Customer value benchmarks
4. **Lifetime Value** - CLV projections by segment
5. **ROI Analysis** - Business impact projections

## 🔧 Technical Architecture

- **Unity Catalog**: Data governance and managed tables
- **Lakeflow (DLT)**: Declarative data pipelines
- **Serverless Compute**: Cost-effective processing
- **Plotly Express**: Accessible, interactive visualizations
- **Synthetic Data**: No external dependencies

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Third-Party Package Licenses

&copy; 2025 Databricks, Inc. All rights reserved. The source in this project is provided subject to the Databricks License [https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses set forth below.

| Package | License | Copyright |
|---------|---------|-----------|
| plotly>=5.15.0 | MIT | Copyright (c) 2016-2023 Plotly, Inc |
| numpy>=1.21.0 | BSD-3-Clause | Copyright (c) 2005-2023, NumPy Developers |
| pandas>=1.5.0 | BSD-3-Clause | Copyright (c) 2008-2023, AQR Capital Management, LLC |
| scikit-learn>=1.3.0 | BSD-3-Clause | Copyright (c) 2007-2023 The scikit-learn developers |
| Faker | MIT | Copyright (c) 2012-2023 joke2k |

## 📜 License

This project is licensed under the Databricks License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects.