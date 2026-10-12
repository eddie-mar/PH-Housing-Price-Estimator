import numpy as np
import pandas as pd

import re

# Area flag for NCR, Near NCR, Cebu. List for cities

NCR = [
    'caloocan',
    'las pinas',
    'makati',
    'malabon',
    'mandaluyong',
    'manila',
    'marikina',
    'muntinlupa',
    'navotas',
    'paranaque',
    'pateros',
    'pasay',
    'pasig',
    'quezon city',
    'san juan',
    'taguig',
    'valenzuela',
]

METRO_CEBU = [
    'cebu',
    'mandaue',
    'lapu-lapu',
    'consolacion',
    'talisay',
    'minglanilla',
    'apas',
    'lahug',
]

NEAR_NCR = [
    # Cavite
    'silang',
    'carmona',
    'alfonso',
    'tagaytay',
    'dasmarinas',
    'bacoor',
    'general trias',
    'imus',
    'trece martires',
    'tanza',
    'kawit',
    'rosario',
    'naic',

    # Laguna
    'los banos',
    'cabuyao',
    'santa rosa',
    'sta rosa city laguna',
    'calamba',
    'binan',
    'san pablo',
    'pagsanjan',
    'pila',
    'lumban',
    'majayjay',
    'calauan',

    # Rizal
    'antipolo',
    'taytay',
    'cainta',
    'san mateo',
    'angono',
    'binangonan',
    'teresa',

    # Bulacan
    'san jose del monte',
    'marilao',
    'pulilan',
    'malolos',
    'baliuag',
    'plaridel',
    'santa maria',
    'meycauayan',
    'pandi',
    'hagonoy',
    'paombong',
    'san rafael',
    'bulacan',
]

# House type
HOUSE_TYPES = ['mansion', 'beach house', 'rest house', 'bungalow', 'townhouse', 'villa', 'rowhouse', 'detached', 'attached', \
            'shophouse', 'duplex', 'triplex', 'condominium']


# Parser functions

def parse_loc(location):
    if pd.isna(location):
        return 'missing', 'missing'
        
    parsed = [part.strip().lower() for part in location.split(',')]
    
    if len(parsed) >= 2:
        locality = parsed[-2]
        city = parsed[-1]
    elif len(parsed) == 1:
        locality = 'missing'
        city = parsed[-1]
    else:
        locality = 'missing'
        city = 'missing'

    return locality, city

def parse_type(description):
    if pd.isna(description):
        return 'others'

    for house_type in HOUSE_TYPES:
        match = re.search(rf'\b({re.escape(house_type)})s?\b', description, flags=re.IGNORECASE)
        if match:
            return house_type.lower()

    return 'others'

def parse_two_storey(description):
    if pd.isna(description):
        return 0

    description = description.lower()
    if re.search(r'2[-\s]?storeys?', description) or re.search(r'two[-\s]?storeys?', description):
        return 1
    else:
        return 0

def parse_three_storey(description):
    if pd.isna(description):
        return 0

    description = description.lower()
    if re.search(r'3[-\s]?storeys?', description) or re.search(r'three[-\s]?storeys?', description):
        return 1
    else:
        return 0

def parse_elite(row):
    if pd.isna(row['description']) or pd.isna(row['location']):
        return 0

    pattern_forbes = r'\bforbes park\b'
    pattern_ayala = r'\bayala alabang\b'

    description = row['description'].lower()
    location = row['location'].lower()

    if re.search(pattern_ayala, location, flags=re.IGNORECASE) or re.search(pattern_ayala, description, flags=re.IGNORECASE):
        return 1
    elif re.search(pattern_forbes, location, flags=re.IGNORECASE) or re.search(pattern_forbes, description, flags=re.IGNORECASE):
        return 1
    else:
        return 0


def det_feature_eng(data):
    # Parse location
    data[['locality', 'city']] = data['location'].apply(lambda x: pd.Series(parse_loc(x)))
    data = data.drop(columns='locality')

    # Flag locality
    data['is_ncr'] = data['city'].isin(NCR).astype(int)
    data['is_metro_cebu'] = data['city'].isin(METRO_CEBU).astype(int)
    data['is_near_ncr'] = data['city'].isin(NEAR_NCR).astype(int)

    # House type    
    data['house_type'] = data['description'].apply(parse_type)

    # Storeys
    data['is_two_storey'] = data['description'].apply(parse_two_storey)
    data['is_three_storey'] = data['description'].apply(parse_three_storey)

    # Elite
    data['is_elite'] = data.apply(parse_elite, axis=1)

    # Ammenities
    def flag_ammenities(prefix, ammenity):
        col_name = f'{prefix}_{ammenity}'
        data[col_name] = np.where(data['description'].str.contains(ammenity, case=False, na=False), 1, 0)

    flag_ammenities('has', 'pool')
    flag_ammenities('is', 'affordable')
    flag_ammenities('is', 'luxury')
    flag_ammenities('is', 'smart')
    flag_ammenities('is', 'overlooking')
    flag_ammenities('is', 'contemporary')
    flag_ammenities('is', 'modern')
    flag_ammenities('is', 'foreclosed')
    flag_ammenities('in', 'subdivision')

    data = data.drop(columns=['description', 'location'])
    # data = pd.get_dummies(data, columns=['house_type'], drop_first=True, dtype=int)
    return data