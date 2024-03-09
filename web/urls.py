from django.urls import path

from web import views

urlpatterns = [
    path("", views.index, name="index"),
    path("about/", views.about, name="about"),
    path("api/classify/", views.classify, name="classify"),
    path("api/history/", views.history, name="history"),
    path("api/analyses/<int:pk>/", views.delete, name="delete"),
    path("api/model/", views.model, name="model"),
    path("healthz/", views.healthz, name="healthz"),
]
