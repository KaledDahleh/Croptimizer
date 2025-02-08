import requests

def fetchWeather(lat, lon):
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date=2024-06-01&end_date=2024-08-27&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,sunshine_duration"
    response = requests.get(url)
    data = response.json()
    return data

def getAvgs(lat, lon):
    data = fetchWeather(lat, lon)
    temps_max = data.get("daily").get("temperature_2m_max")
    temps_min = data.get("daily").get("temperature_2m_min")
    precip = data.get("daily").get("precipitation_sum")
    wind_speed = data.get("daily").get("wind_speed_10m_max")
    sunshine = data.get("daily").get("sunshine_duration")

    avg_temp_max = sum(temps_max) / len(temps_max)
    avg_temp_min = sum(temps_min) / len(temps_min)
    avg_temp = (sum(temps_max) + sum(temps_min)) / (2 * len(temps_max))
    avg_precip = sum(precip) / len(precip)
    avg_wind_speed = sum(wind_speed) / len(wind_speed)
    avg_sunshine = sum(sunshine) / len(sunshine) / 60 / 60

    avg_humidity = min(100, max(30, (avg_precip * 1.2 + 60 - avg_temp * 0.5)))

    return dict(avg_temp_max=avg_temp_max, avg_temp_min=avg_temp_min, avg_temp=avg_temp, avg_precip=avg_precip, avg_wind_speed=avg_wind_speed, avg_sunshine=avg_sunshine, avg_humidity=avg_humidity)

if __name__ == "__main__":
    print(getAvgs(67.7749, 22.4194))
