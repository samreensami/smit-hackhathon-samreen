# src/utils.py

# Major Pakistani cities as per user request
CITIES = ["Lahore", "Karachi", "Islamabad", "Peshawar"]

# Dictionary mapping cities to numeric codes and vice versa
CITY_TO_CODE = {city: i for i, city in enumerate(CITIES)}
CODE_TO_CITY = {i: city for i, city in enumerate(CITIES)}

def get_aqi_category(aqi_value): # Renamed from pm25_value to aqi_value for generality
    """
    Returns AQI category based on a generic AQI value (e.g., CO_AQI) as per Pakistan EPA standards (simplified).
    Ranges provided by the user:
    'Good': 0-50
    'Moderate': 51-100
    'Unhealthy': 101-250
    'Hazardous': 251+
    """
    if aqi_value is None:
        return "Unknown"
    
    # Ensure aqi_value is a number for comparison
    try:
        aqi_value = float(aqi_value)
    except (ValueError, TypeError):
        return "Unknown"

    if 0 <= aqi_value <= 50:
        return "Good"
    elif 51 <= aqi_value <= 100:
        return "Moderate"
    elif 101 <= aqi_value <= 250:
        return "Unhealthy"
    elif aqi_value >= 251:
        return "Hazardous"
    else: # For negative or abnormally high values not covered by the above
        return "Unknown"
