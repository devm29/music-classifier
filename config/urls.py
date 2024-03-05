from django.conf import settings
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("web.urls")),
]

if settings.SERVE_MEDIA:
    from django.views.static import serve

    urlpatterns += [
        path("media/<path:path>", serve, {"document_root": settings.MEDIA_ROOT}),
    ]
