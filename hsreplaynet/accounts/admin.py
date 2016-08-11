from django.contrib import admin
from .models import AccountClaim, User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin


@admin.register(AccountClaim)
class AccountClaimAdmin(admin.ModelAdmin):
	list_display = ("__str__", "id", "token", "created")
	raw_id_fields = ("token", )


@admin.register(User)
class UserAdmin(BaseUserAdmin):
	change_form_template = "loginas/change_form.html"
	list_display = ("username", "date_joined", "last_login", "delete_account_request")
	list_filter = BaseUserAdmin.list_filter + ("is_fake", )
