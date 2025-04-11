import googlemaps
import streamlit as st
import pandas as pd
import time
import math
import logging
from geopy.exc import GeocoderTimedOut

# Set up logging to write logs to a file
logging.basicConfig(filename="geocoding_logs.txt", level=logging.INFO)

# Function to calculate the Haversine distance between two coordinates
def haversineDistance(coords1, coords2):
    R = 6371  # Radius of Earth in km
    lat1 = coords1[0] * (math.pi / 180)
    lat2 = coords2[0] * (math.pi / 180)
    deltaLat = (coords2[0] - coords1[0]) * (math.pi / 180)
    deltaLng = (coords2[1] - coords1[1]) * (math.pi / 180)

    a = math.sin(deltaLat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(deltaLng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in km

# Function to get latitude and longitude with retries (Handles timeouts and retries)
def get_lat_long_with_retry(address, gmaps, retries=3, delay=2):
    for attempt in range(retries):
        try:
            # Google Maps Geocoding request
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]["geometry"]["location"]
                return (location["lat"], location["lng"])
            else:
                logging.warning(f"Geocoding failed for address: {address}")
                return None
        except GeocoderTimedOut:
            logging.warning(f"Geocoding timed out for address: {address}")
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            logging.error(f"Error geocoding address: {address} - {e}")
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    return None  # Return None if all retries fail

# Function to find the closest addresses
def find_closest_addresses(addresses, gmaps, progress_bar, progress_text):
    results = []
    num_addresses = len(addresses)  # Track the number of addresses to be processed
    for i, origin in enumerate(addresses):
        origin_coords = get_lat_long_with_retry(origin, gmaps)
        closest_address = ''
        closest_distance = float('inf')  # Initialize to a large number

        if origin_coords:
            for j, destination in enumerate(addresses):
                if i != j:  # Don't compare the address with itself
                    destination_coords = get_lat_long_with_retry(destination, gmaps)

                    # Only calculate distance if both coordinates are available
                    if destination_coords:
                        distance = haversineDistance(origin_coords, destination_coords)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_address = destination
            if closest_address:
                results.append(f"{closest_address} ({closest_distance:.2f} km)")
            else:
                results.append("No close address found")
        else:
            results.append(f"Geocoding error for origin: {origin}")

        # Update the progress bar with "1 of 10" format
        progress_bar.progress((i + 1) / num_addresses)
        progress_text.text(f"Processing {i + 1} of {num_addresses}")

    return results

# Streamlit interface
st.title("Address Distance Finder")

# Upload file (CSV format)
uploaded_file = st.file_uploader("Upload your CSV file with addresses", type=["csv"])

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:", df.head())

    # Get the list of addresses (assuming addresses are in the first column)
    addresses = df.iloc[:, 0].dropna().tolist()

    # Show the progress text at the top
    progress_text = st.empty()  # This will dynamically update the text
    progress_text.text("Processing 0 of 0")  # Initial placeholder text

    # Show the progress bar at the top
    progress_bar = st.progress(0)

    # Button to run the calculation
    if st.button('Find Closest Addresses'):
        if len(addresses) > 0:
            # Initialize Google Maps client with secret key
            try:
                gmaps = googlemaps.Client(key=st.secrets["google_maps"]["google_maps_key"])  # Ensure correct access syntax
                logging.info("Google Maps API client initialized successfully.")
            except Exception as e:
                st.error(f"Error initializing Google Maps client: {e}")
                logging.error(f"Error initializing Google Maps client: {e}")
                st.stop()

            # Find the closest addresses
            results = find_closest_addresses(addresses, gmaps, progress_bar, progress_text)

            # Add results to the dataframe (in columns for closest address and distance)
            df['Closest Address'] = [result.split(' (')[0] for result in results]
            df['Distance (km)'] = [result.split('(')[-1].replace(')', '').strip() if '(' in result else 'N/A' for result in results]

            # Display results
            st.write("Results:", df)

            # Download button for the results
            csv = df.to_csv(index=False)
            st.download_button("Download results as CSV", csv, "results.csv", "text/csv")

            # Notify the user that the process is done
            st.success("Geocoding and distance calculations completed!")

        else:
            st.error("Please upload a file with addresses.")
