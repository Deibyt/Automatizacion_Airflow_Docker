from django.urls import path
from .views import SignalResultsAPIView

urlpatterns = [
    path('signals/', SignalResultsAPIView.as_view(), name='signal-results'),
]
