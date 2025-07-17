# Customer Segmentation Solution Accelerator

![Customer Segmentation](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)
![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)

Modern customer segmentation using Databricks Unity Catalog, Lakeflow Declarative Pipelines, and Serverless Compute. This solution demonstrates best practices for RFM analysis combined with behavioral clustering to create actionable customer segments.

## Quick Start

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- Databricks CLI installed and configured
- Cluster creation permissions

### Deploy
```bash
git clone https://github.com/databricks-industry-solutions/segmentation.git
cd segmentation
./scripts/deploy.sh
```

## What's Included

- **Synthetic Data Generation**: 10K customers, 250K transactions
- **RFM + Behavioral Analysis**: Advanced customer segmentation
- **Interactive Visualizations**: Plotly dashboards with business insights
- **Modern Architecture**: DAB format, Unity Catalog, Serverless Compute

## Architecture

```
Data Generation → Unity Catalog → RFM Analysis → K-Means Clustering → Business Insights
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_Data_Generation` | Creates synthetic customer data |
| `02_Segmentation_Analysis` | RFM + behavioral clustering |
| `03_Business_Insights` | Interactive Plotly visualizations |

## Customer Segments

1. **High-Value Loyalists** - Premium customers, VIP programs
2. **Frequent Shoppers** - Regular customers, loyalty rewards  
3. **Discount Hunters** - Price-sensitive, strategic promotions
4. **Occasional Buyers** - Sporadic purchasers, engagement campaigns
5. **New/Inactive Customers** - Reactivation strategies
6. **Category Specialists** - Cross-selling opportunities

## Business Impact

- **20% projected revenue lift** through targeted segmentation
- **Improved customer lifetime value** across segments
- **Enhanced marketing efficiency** with precise targeting

## Configuration

Update `databricks.yml` for your environment:
```yaml
variables:
  catalog: your_catalog_name
  schema: your_schema_name
```

## Cleanup
```bash
./scripts/cleanup.sh
```

## Third-Party Package Licenses

&copy; 2025 Databricks, Inc. All rights reserved. The source in this project is provided subject to the Databricks License [https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses set forth below.

| Package | License | Copyright |
|---------|---------|-----------|
| plotly>=5.15.0 | MIT | Copyright (c) 2016-2023 Plotly, Inc |
| numpy>=1.21.0 | BSD-3-Clause | Copyright (c) 2005-2023, NumPy Developers |
| pandas>=1.5.0 | BSD-3-Clause | Copyright (c) 2008-2023, AQR Capital Management, LLC |
| scikit-learn>=1.3.0 | BSD-3-Clause | Copyright (c) 2007-2023 The scikit-learn developers |
| Faker | MIT | Copyright (c) 2012-2023 joke2k |

## License

This project is licensed under the Databricks License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects.