# crypto-market-intelligence-platform

A real-world end-to-end batch data engineering project designed to collect, process, transform, and analyze cryptocurrency market data using the **CoinGecko REST API, Python, PySpark, Apache Airflow, AWS, Amazon Redshift, and Power BI**.

This project demonstrates a **complete data engineering lifecycle** — from **API-based data extraction and raw data lake storage** to **distributed data processing, data validation, feature engineering, data warehousing, pipeline orchestration, CI/CD, monitoring, and business intelligence reporting.**

# Project Overview

* Cryptocurrency markets generate large volumes of continuously changing market data, including prices, market capitalization, trading volume, supply, and historical market indicators.
* This project automates the process of collecting cryptocurrency market data from the **CoinGecko REST API**, validating and storing the raw data in **Amazon S3**, processing it using **PySpark on Amazon EMR**, and loading analytics-ready data into **Amazon Redshift Serverless**.
* The pipeline is orchestrated using **Apache Airflow on Amazon MWAA**, which manages the complete ETL workflow and Amazon EMR cluster lifecycle.
* The processed data is stored in **Parquet format** and made available for SQL analytics and **Power BI dashboard reporting**.
* The project also includes **GitHub Actions CI/CD**, automated testing, code quality checks, AWS OIDC authentication, centralized logging, and Amazon SNS notifications.

# Project Objectives

The primary objectives of this project include:

* Extract cryptocurrency market data from the **CoinGecko REST API**
* Perform data validation during the extraction process
* Store raw API responses in **Amazon S3**
* Design a structured S3 data lake with raw and processed data layers
* Process large datasets using **Apache Spark and PySpark**
* Perform schema management and data quality validation
* Apply data transformations and feature engineering
* Generate analytics-ready **Parquet datasets**
* Orchestrate the complete ETL workflow using **Apache Airflow on Amazon MWAA**
* Automate the **Amazon EMR cluster lifecycle**
* Load processed data into **Amazon Redshift Serverless**
* Enable SQL-based analytical reporting
* Build interactive **Power BI dashboards**
* Implement automated CI validation using **GitHub Actions**
* Implement secure CD deployment using **AWS IAM OIDC authentication**
* Integrate logging, callbacks, and **Amazon SNS notifications** for pipeline monitoring
* Deploy the pipeline using secure AWS networking and IAM architecture

# Data Engineering Workflow
```
CoinGecko REST API
↓
Python API Extraction & Validation
↓
Amazon S3 – Raw Data Layer
↓
Apache Airflow / Amazon MWAA
↓
Amazon EMR – PySpark Processing
↓
Data Validation & Feature Engineering
↓
Parquet – Processed Data Layer
↓
Amazon Redshift Serverless
↓
SQL Analytics
↓
Power BI Dashboards
```

#  AWS Architecture

The project uses the following AWS services:

* **Amazon S3** – Data lake for raw and processed datasets
* **Amazon EMR** – Distributed PySpark data processing
* **Amazon MWAA** – Managed Apache Airflow orchestration
* **Amazon Redshift Serverless** – Analytical data warehouse
* **Amazon VPC** – Network isolation
* **Public & Private Subnets** – Network segmentation
* **Internet Gateway** – Internet connectivity for public resources
* **NAT Gateway** – Outbound internet access for private resources
* **Route Tables** – Network traffic routing
* **Security Groups** – Network-level access control
* **IAM Roles & Policies** – Secure AWS service permissions
* **Amazon SNS** – Pipeline notifications and alerts

# Data Processing

The PySpark processing workflow performs:

* Schema management
* Data type handling
* Data validation
* Data quality checks
* Data transformation
* Feature engineering
* Missing and invalid data handling
* Conversion of processed datasets into **Parquet format**
* Preparation of analytics-ready datasets for Amazon Redshift

#  Feature Engineering

The pipeline generates additional analytical features such as:

* Days Since All-Time High
* Days Since All-Time Low
* Price Volatility
* Distance from All-Time High
* Distance from All-Time Low
* Volume-to-Market-Cap Ratio
* Supply Utilization
* Price Direction

These features help provide additional insights into cryptocurrency market behavior and trends.

#  Technology Stack

### Programming & Processing

* Python
* SQL
* Apache Spark
* PySpark
* Pandas

### AWS

* Amazon S3
* Amazon EMR
* Amazon MWAA
* Amazon Redshift Serverless
* Amazon VPC
* IAM
* Amazon SNS
* AWS Secrets Manager

### Orchestration

* Apache Airflow
* Amazon MWAA

### DevOps & Testing

* Git
* GitHub
* GitHub Actions
* AWS IAM OIDC
* Ruff
* Black
* Mypy
* Pytest
* AWS CLI

### Analytics & Visualization

* Power BI

#  Airflow & EMR Orchestration

Apache Airflow on **Amazon MWAA** orchestrates the complete ETL workflow.

The DAG manages the Amazon EMR lifecycle through the following stages:

* Create EMR cluster
* Monitor cluster readiness
* Submit PySpark transformation job
* Monitor Spark job completion
* Terminate EMR cluster

This approach allows the processing environment to be created when required and terminated after processing is completed.

#  CI/CD Pipeline

The project uses **GitHub Actions** to automate code validation and deployment.

### CI Pipeline

The CI workflow performs:

* Ruff code quality checks
* Black formatting validation
* Mypy static type checking
* Pytest unit tests
* Pytest integration tests
* Test coverage validation
* Configuration validation
* Airflow DAG validation
* YAML validation
* DAG compilation/import checks
* Package build validation

### CD Pipeline

The deployment workflow uses **AWS IAM OIDC authentication** instead of long-lived AWS access keys.

The CD process deploys:

* Airflow DAGs
* Plugins
* Dependency requirements

to **Amazon MWAA**.

#  AWS Security & Networking

The project implements a secure AWS architecture using:

* Amazon VPC
* Public and private subnets
* Route tables
* Internet Gateway
* NAT Gateway
* Security Groups
* IAM Roles and Policies
* IAM OIDC authentication for GitHub Actions

The architecture separates network resources and controls service-to-service access through IAM permissions and security groups.

#  Monitoring & Notifications

The pipeline includes centralized monitoring and notification mechanisms:

* Airflow task logging
* Pipeline callbacks
* Success notifications
* Failure notifications
* Amazon SNS alerts
* Environment-based configuration

This helps identify pipeline failures and monitor ETL execution.

#  Power BI Analytics

The processed cryptocurrency data is connected to **Power BI** to create interactive dashboards for analyzing:

* Cryptocurrency prices
* Market capitalization
* Trading volume
* Price trends
* Market performance
* Cryptocurrency market indicators

The dashboards transform the processed warehouse data into interactive visual insights.

#  Key Project Outcomes

This project demonstrates practical experience in:

* Building end-to-end batch ETL pipelines
* REST API data extraction
* Data lake architecture
* Distributed data processing with PySpark
* Cloud-based pipeline orchestration
* Amazon EMR lifecycle management
* Data warehousing with Redshift Serverless
* AWS networking and IAM
* Automated testing and code quality
* CI/CD with GitHub Actions
* Secure OIDC-based AWS deployment
* Pipeline monitoring and notifications
* Business intelligence using Power BI

