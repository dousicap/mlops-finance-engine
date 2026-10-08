# Copier en terraform.tfvars (ignoré par Git) et adapter.
github_repository  = "dousicap/mlops-finance-engine"
region             = "eu-west-3"
enable_nat_gateway = false # démo à moindre coût. true en production.
desired_count      = 1
