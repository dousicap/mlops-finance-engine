output "api_url" {
  value = "${local.https ? "https" : "http"}://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "github_deploy_role_arn" {
  description = "À enregistrer dans GitHub : Settings > Secrets and variables > Actions > Variables > AWS_DEPLOY_ROLE_ARN"
  value       = aws_iam_role.github_deploy.arn
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "deployment_strategy" {
  value = var.deployment_strategy
}

output "codedeploy_application" {
  value = local.blue_green ? aws_codedeploy_app.api[0].name : null
}

output "codedeploy_deployment_group" {
  value = local.blue_green ? aws_codedeploy_deployment_group.api[0].deployment_group_name : null
}
