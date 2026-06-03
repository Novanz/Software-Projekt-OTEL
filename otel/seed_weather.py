import os
from collections import defaultdict

import chromadb
import requests

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "weather_rag_docs")
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "true").lower() == "true"

OPEN_METEO_ARCHIVE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)
START_DATE = os.getenv("START_DATE", "2025-03-01")
END_DATE = os.getenv("END_DATE", "2025-04-30")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")

CITIES = [
    {"id": "frankfurt", "name": "Frankfurt am Main", "latitude": 50.1109, "longitude": 8.6821},
    {"id": "berlin", "name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
    {"id": "hamburg", "name": "Hamburg", "latitude": 53.5511, "longitude": 9.9937},
    {"id": "munich", "name": "Munich", "latitude": 48.1374, "longitude": 11.5755},
]

DAILY_FIELDS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
]

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}

def fetch_city_daily(city: dict) -> dict:
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_FIELDS),
        "timezone": TIMEZONE,
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["daily"]

def build_documents(city: dict, daily: dict) -> list[dict]:
    times = daily["time"]
    docs = []

    for i in range(0, len(times), 7):
        week_slice = slice(i, i + 7)
        dates = times[week_slice]
        if not dates:
            continue

        tmax = daily["temperature_2m_max"][week_slice]
        tmin = daily["temperature_2m_min"][week_slice]
        tmean = daily["temperature_2m_mean"][week_slice]
        precip = daily["precipitation_sum"][week_slice]
        rain = daily["rain_sum"][week_slice]
        wind = daily["wind_speed_10m_max"][week_slice]
        codes = daily["weather_code"][week_slice]

        avg_mean_temp = round(sum(v for v in tmean if v is not None) / max(1, len([v for v in tmean if v is not None])), 1)
        max_temp = round(max(v for v in tmax if v is not None), 1)
        min_temp = round(min(v for v in tmin if v is not None), 1)
        total_precip = round(sum(v for v in precip if v is not None), 1)
        total_rain = round(sum(v for v in rain if v is not None), 1)
        max_wind = round(max(v for v in wind if v is not None), 1)

        conditions = defaultdict(int)
        for code in codes:
            conditions[WEATHER_CODE_MAP.get(code, f"Code {code}")] += 1
        top_conditions = ", ".join(
            f"{label} ({count}d)"
            for label, count in sorted(conditions.items(), key=lambda x: (-x[1], x[0]))[:3]
        )

        week_start = dates[0]
        week_end = dates[-1]
        doc_id = f"{city['id']}-{week_start}"
        title = f"Weather summary for {city['name']} from {week_start} to {week_end}"

        daily_lines = []
        for d, hi, lo, mean, pr, rn, ws, wc in zip(dates, tmax, tmin, tmean, precip, rain, wind, codes):
            daily_lines.append(
                f"- {d}: mean {mean}C, high {hi}C, low {lo}C, precipitation {pr}mm, rain {rn}mm, max wind {ws} km/h, condition {WEATHER_CODE_MAP.get(wc, wc)}."
            )

        text = "\n".join(
            [
                f"City: {city['name']}",
                f"Period: {week_start} to {week_end}",
                f"Average mean temperature: {avg_mean_temp}C",
                f"Highest daily maximum temperature: {max_temp}C",
                f"Lowest daily minimum temperature: {min_temp}C",
                f"Total precipitation: {total_precip}mm",
                f"Total rain: {total_rain}mm",
                f"Maximum wind speed: {max_wind} km/h",
                f"Most common conditions: {top_conditions}",
                "Daily breakdown:",
                *daily_lines,
            ]
        )

        docs.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "title": title,
                    "city": city["name"],
                    "city_id": city["id"],
                    "country": "DE",
                    "week_start": week_start,
                    "week_end": week_end,
                    "source": "open-meteo-archive",
                    "dataset_type": "weather_weekly_summary",
                },
            }
        )

    return docs

def main():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    all_docs = []
    for city in CITIES:
        daily = fetch_city_daily(city)
        all_docs.extend(build_documents(city, daily))

    collection.add(
        ids=[d["id"] for d in all_docs],
        documents=[f"{d['title']}\n{d['text']}" for d in all_docs],
        metadatas=[d["metadata"] for d in all_docs],
    )

    print(f"Seeded {len(all_docs)} docs into collection '{COLLECTION_NAME}'")
    print(f"Collection count: {collection.count()}")

if __name__ == "__main__":
    main()
