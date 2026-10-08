# Déploiement blue/green (deployment_strategy = "blue_green").
#
# Déroulé : CodeDeploy démarre la version green dans le second target group, l'expose sur le
# listener de test (:9000, réservé à l'équipe), puis bascule le trafic de production en canary
# (10 % pendant 5 min, puis 100 %). Pendant la bascule, toute alarme ci-dessous déclenche le
# retour arrière automatique vers blue, qui reste disponible 15 min après la fin.

locals {
  blue_green = var.deployment_strategy == "blue_green"
}

resource "aws_lb_target_group" "api_green" {
  count                = local.blue_green ? 1 : 0
  name                 = "${var.project}-api-green"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "test" {
  count             = local.blue_green ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 9000
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_green[0].arn
  }

  lifecycle {
    ignore_changes = [default_action] # CodeDeploy alterne blue et green
  }
}

# Alarmes de déploiement : elles portent sur ce que subit l'utilisateur (tout l'ALB),
# pas sur un target group précis qui change de rôle à chaque bascule.
resource "aws_cloudwatch_metric_alarm" "deploy_5xx" {
  count               = local.blue_green ? 1 : 0
  alarm_name          = "${var.project}-deploy-5xx"
  alarm_description   = "Erreurs 5xx de l'API pendant un déploiement : retour arrière"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 5
  period              = 60
  evaluation_periods  = 2
  treat_missing_data  = "notBreaching"
  dimensions          = { LoadBalancer = aws_lb.main.arn_suffix }
}

resource "aws_cloudwatch_metric_alarm" "deploy_latency" {
  count               = local.blue_green ? 1 : 0
  alarm_name          = "${var.project}-deploy-latency-p95"
  alarm_description   = "Latence p95 supérieure à 1 s pendant un déploiement : retour arrière"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 3
  treat_missing_data  = "notBreaching"
  dimensions          = { LoadBalancer = aws_lb.main.arn_suffix }
}

data "aws_iam_policy_document" "codedeploy_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codedeploy.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "codedeploy" {
  count              = local.blue_green ? 1 : 0
  name               = "${var.project}-codedeploy"
  assume_role_policy = data.aws_iam_policy_document.codedeploy_assume.json
}

resource "aws_iam_role_policy_attachment" "codedeploy" {
  count      = local.blue_green ? 1 : 0
  role       = aws_iam_role.codedeploy[0].name
  policy_arn = "arn:aws:iam::aws:policy/AWSCodeDeployRoleForECS"
}

resource "aws_codedeploy_app" "api" {
  count            = local.blue_green ? 1 : 0
  name             = "${var.project}-api"
  compute_platform = "ECS"
}

resource "aws_codedeploy_deployment_group" "api" {
  count                  = local.blue_green ? 1 : 0
  app_name               = aws_codedeploy_app.api[0].name
  deployment_group_name  = "${var.project}-api"
  service_role_arn       = aws_iam_role.codedeploy[0].arn
  deployment_config_name = var.codedeploy_config

  deployment_style {
    deployment_type   = "BLUE_GREEN"
    deployment_option = "WITH_TRAFFIC_CONTROL"
  }

  ecs_service {
    cluster_name = aws_ecs_cluster.main.name
    service_name = aws_ecs_service.api.name
  }

  blue_green_deployment_config {
    deployment_ready_option {
      action_on_timeout = "CONTINUE_DEPLOYMENT"
    }
    terminate_blue_instances_on_deployment_success {
      action                           = "TERMINATE"
      termination_wait_time_in_minutes = 15
    }
  }

  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE", "DEPLOYMENT_STOP_ON_ALARM"]
  }

  alarm_configuration {
    enabled = true
    alarms = [
      aws_cloudwatch_metric_alarm.deploy_5xx[0].alarm_name,
      aws_cloudwatch_metric_alarm.deploy_latency[0].alarm_name,
    ]
  }

  load_balancer_info {
    target_group_pair_info {
      prod_traffic_route {
        listener_arns = [local.https ? aws_lb_listener.https[0].arn : aws_lb_listener.http.arn]
      }
      test_traffic_route {
        listener_arns = [aws_lb_listener.test[0].arn]
      }
      target_group {
        name = aws_lb_target_group.api.name
      }
      target_group {
        name = aws_lb_target_group.api_green[0].name
      }
    }
  }
}
