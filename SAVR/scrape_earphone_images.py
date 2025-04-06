import os
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests

def clean_filename(filename):
    # Remove invalid characters from filename
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def download_image(url, filename):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded: {filename}")
            return True
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_first_image_url(driver, search_query):
    try:
        # Construct the search URL
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        
        # Load the page
        driver.get(search_url)
        
        # Wait for images to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "img.rg_i")))
        
        # Find the first image
        images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
        if images:
            # Click the first image to get the full resolution version
            images[0].click()
            time.sleep(2)  # Wait for the full resolution image to load
            
            # Find the full resolution image
            full_res_images = driver.find_elements(By.CSS_SELECTOR, "img.r48jcc")
            if full_res_images:
                return full_res_images[0].get_attribute('src')
        
        print(f"No image found for: {search_query}")
        return None
    except Exception as e:
        print(f"Error getting image URL for {search_query}: {str(e)}")
        return None

def main():
    # List of search queries
    search_queries = [
        "Apple AirPods Pro (2nd Gen)",
        "Sony WF-1000XM5",
        "Bose QuietComfort Earbuds II",
        "Samsung Galaxy Buds 2 Pro",
        "Jabra Elite 7 Pro",
        "Sennheiser Momentum True Wireless 3",
        "Google Pixel Buds Pro",
        "OnePlus Buds Pro 2",
        "Nothing Ear (2)",
        "Beats Fit Pro",
        "Sony LinkBuds S",
        "Anker Soundcore Liberty 4",
        "Realme Buds Air 3",
        "Redmi Buds 4 Pro",
        "Oppo Enco X2",
        "JBL Live Pro 2",
        "Skullcandy Indy ANC",
        "SoundPEATS Air3 Pro",
        "Huawei FreeBuds Pro 2",
        "Boat Airdopes 441",
        "Noise Air Buds Pro 2",
        "Boult Audio AirBass XPods Pro",
        "Wings Phantom 850",
        "Ptron Bassbuds Jade",
        "Mivi DuoPods A350",
        "Fire-Boltt FirePods Polaris",
        "Hammer Airflow Pro",
        "Dizo Buds Z Pro",
        "Wings Phantom 550"
    ]

    # Create the directory if it doesn't exist
    output_dir = "earphone images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup the driver
    driver = setup_driver()

    try:
        # Process each search query
        for query in search_queries:
            print(f"\nProcessing: {query}")
            
            # Get the first image URL
            image_url = get_first_image_url(driver, query)
            
            if image_url:
                # Create a clean filename from the query
                filename = clean_filename(query) + ".jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Download the image
                download_image(image_url, filepath)
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(2)
            else:
                print(f"Skipping {query} due to no image found")
    
    finally:
        # Always close the driver
        driver.quit()

if __name__ == "__main__":
    main() 