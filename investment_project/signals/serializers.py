from rest_framework import serializers

class SignalResultSerializer(serializers.Serializer):
    ticker = serializers.CharField()
    original_signal = serializers.CharField()
    RandomForest_prediction = serializers.CharField()
    LogisticRegression_prediction = serializers.CharField()
    SVC_prediction = serializers.CharField()
