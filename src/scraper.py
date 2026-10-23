import os
import string
import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_ufc_fighters():
    print("--- Scraping Live Fighter Stats from UFCStats.com ---")
    headers = {'User-Agent': 'Mozilla/5.0'}
    fighters = {}

    # Iterate through the UFCStats alphabet directory (A-Z)
    for letter in string.ascii_lowercase:
        url = f"http://ufcstats.com/statistics/fighters?char={letter}&page=all"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            continue
            
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.find_all('tr', class_='b-statistics__table-row')
        
        for row in rows[2:]:  # Skip header rows
            cols = [td.text.strip() for td in row.find_all('td')]
            if len(cols) >= 11:
                first_name = cols[0]
                last_name = cols[1]
                full_name = f"{first_name} {last_name}".strip().lower()
                
                # Parse height string (e.g. 5' 11") into centimeters
                height_str = cols[2].replace('"', '').strip()
                height_cms = 175.0
                if "'" in height_str:
                    try:
                        parts = height_str.split("'")
                        feet = float(parts[0].strip())
                        inches = float(parts[1].strip()) if parts[1].strip() else 0.0
                        height_cms = (feet * 12 + inches) * 2.54
                    except ValueError:
                        pass
                
                # Parse reach string (e.g. 74") into centimeters
                reach_str = cols[4].replace('"', '').strip()
                reach_cms = 175.0
                if reach_str.isdigit():
                    reach_cms = float(reach_str) * 2.54
                
                # Parse career wins/losses
                wins = int(cols[6]) if cols[6].isdigit() else 0
                losses = int(cols[7]) if cols[7].isdigit() else 0
                
                fighters[full_name] = {
                    'height_cms': height_cms,
                    'reach_cms': reach_cms,
                    'wins': wins,
                    'losses': losses
                }

    print(f"✅ Successfully scraped live stats for {len(fighters)} fighters!")
    return fighters

def update_master_csv(scraped_fighters):
    csv_path = os.path.join("data", "raw", "ufc-master.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Master CSV not found at {csv_path}")
        return

    print("--- Updating data/raw/ufc-master.csv ---")
    df = pd.read_csv(csv_path)
    updated_count = 0

    # Iterate through dataframe and update fighter physicals and records
    for idx, row in df.iterrows():
        r_name = str(row.get('R_fighter', '')).lower()
        b_name = str(row.get('B_fighter', '')).lower()

        # Update Red Corner stats if present
        if r_name in scraped_fighters:
            stats = scraped_fighters[r_name]
            if 'R_height_cms' in df.columns:
                df.at[idx, 'R_height_cms'] = stats['height_cms']
            if 'R_reach_cms' in df.columns:
                df.at[idx, 'R_reach_cms'] = stats['reach_cms']
            if 'R_losses' in df.columns:
                df.at[idx, 'R_losses'] = stats['losses']
            updated_count += 1

        # Update Blue Corner stats if present
        if b_name in scraped_fighters:
            stats = scraped_fighters[b_name]
            if 'B_height_cms' in df.columns:
                df.at[idx, 'B_height_cms'] = stats['height_cms']
            if 'B_reach_cms' in df.columns:
                df.at[idx, 'B_reach_cms'] = stats['reach_cms']
            if 'B_losses' in df.columns:
                df.at[idx, 'B_losses'] = stats['losses']
            updated_count += 1

    # Overwrite the updated dataframe back to CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully refreshed active fighter profiles across master dataset!")

if __name__ == "__main__":
    scraped_data = scrape_ufc_fighters()
    update_master_csv(scraped_data)