import pandas as pd
import requests

print("Hello")
TIMESERIESURL = "https://hhnk.lizard.net/api/v4/timeseries/{}/events/"

start_date = "2025-08-01"
end_date = "2025-08-05"
start = f"{start_date}T00:00:00Z"
end = f"{end_date}T23:59:59Z"

headers = {
            "Content-Type": "application/json",
        }

uuids = ["6a71ab24-bac4-4f1d-8cc3-47f70cd3066c", "902a3362-7ef6-472e-92de-8a734743e86f"]

all_series_list = []

for uuid in uuids:
    url = TIMESERIESURL.format(uuid)
    response = requests.get(
        url=url,
        headers=headers,
        params={
            'start': start,
            'end': end,
            'fields': 'value',
            'page_size': '10000000'
        }
    )
    response.raise_for_status()

    time_series_events = pd.DataFrame(response.json()['results'])

    # Verwerk timestamps
    time_series_events['time'] = time_series_events['time'].str.replace(r'\.\d+Z', 'Z', regex=True)
    time_series_events['timestamp'] = pd.to_datetime(time_series_events['time'])

    # Maak kolomnaam
    series_name = uuid

    # Bereid data voor
    time_series_events = time_series_events[['timestamp', 'value']]
    time_series_events = time_series_events.rename(columns={'value': series_name})
    time_series_events['timestamp'] = time_series_events['timestamp'].dt.round('min')
    time_series_events = time_series_events.set_index('timestamp')

    all_series_list.append(time_series_events)

# Combineer alle series
if all_series_list:
    all_series_list = [df.reset_index().drop_duplicates(subset='timestamp').set_index('timestamp') 
                        for df in all_series_list]
    merged_df = pd.concat(all_series_list, axis=1)

print(url)
print(merged_df)
