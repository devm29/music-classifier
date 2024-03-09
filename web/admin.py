from django.contrib import admin

from web.models import Analysis


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("original_name", "top_genre", "top_probability", "band", "created_at")
    list_filter = ("top_genre", "band")
    search_fields = ("original_name", "sha256")
    readonly_fields = ("sha256", "result", "created_at")
