variable "project" {
  description = "Préfixe des ressources"
  type        = string
  default     = "finml"
}

variable "region" {
  description = "Région AWS (Paris par défaut : données hébergées en France)"
  type        = string
  default     = "eu-west-3"
}

variable "github_repository" {
  description = "Dépôt autorisé à déployer via OIDC, au format proprietaire/nom"
  type        = string
}

variable "image_tag" {
  description = "Tag de l'image déployée à la création. Ensuite, c'est la CI qui publie les nouvelles révisions."
  type        = string
  default     = "bootstrap"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "enable_nat_gateway" {
  description = "true : conteneurs en sous-réseaux privés derrière une NAT (production, ~35 $/mois). false : sous-réseaux publics (démo)."
  type        = bool
  default     = true
}

variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "max_count" {
  type    = number
  default = 3
}

variable "certificate_arn" {
  description = "Certificat ACM pour HTTPS. Vide : écoute HTTP seulement (démo)."
  type        = string
  default     = ""
}

variable "secret_env" {
  description = "Variables d'environnement lues dans AWS Secrets Manager : { NOM_VARIABLE = arn_du_secret }"
  type        = map(string)
  default     = {}
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "deployment_strategy" {
  description = "rolling : remplacement progressif avec circuit breaker ECS. blue_green : CodeDeploy, bascule progressive du trafic et retour arrière sur alarme."
  type        = string
  default     = "rolling"
  validation {
    condition     = contains(["rolling", "blue_green"], var.deployment_strategy)
    error_message = "deployment_strategy vaut rolling ou blue_green."
  }
}

variable "codedeploy_config" {
  description = "Rythme de bascule en blue/green. Canary : 10 % du trafic pendant 5 min, puis 100 %."
  type        = string
  default     = "CodeDeployDefault.ECSCanary10Percent5Minutes"
}

variable "test_listener_cidrs" {
  description = "Adresses autorisées sur le listener de test (port 9000) qui expose la version green avant bascule. Jamais 0.0.0.0/0."
  type        = list(string)
  default     = []
}
