"""
Amateur Radio Callsign Generator

Generates realistic amateur radio callsigns following international conventions:
- Country prefixes
- Proper number and letter combinations
- Special event and contest callsigns
- Portable and mobile indicators
"""

import random
from typing import List, Optional, Tuple


class CallsignGenerator:
    """Generate realistic amateur radio callsigns."""
    
    # Country prefixes (common ones for contests)
    COUNTRY_PREFIXES = {
        'USA': {
            'prefixes': ['W', 'K', 'N', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 
                        'AI', 'AJ', 'AK', 'KA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG',
                        'KI', 'KJ', 'KK', 'KM', 'KN', 'KO', 'KP', 'KQ', 'KR', 'KS',
                        'KT', 'KU', 'KV', 'KW', 'KX', 'KY', 'KZ', 'NA', 'NB', 'NC',
                        'ND', 'NE', 'NF', 'NG', 'NI', 'NJ', 'NK', 'NL', 'NM', 'NN',
                        'NO', 'NP', 'NQ', 'NR', 'NS', 'NT', 'NU', 'NV', 'NW', 'NX',
                        'NY', 'NZ', 'WA', 'WB', 'WC', 'WD', 'WE', 'WF', 'WG', 'WI',
                        'WJ', 'WK', 'WL', 'WM', 'WN', 'WO', 'WP', 'WQ', 'WR', 'WS',
                        'WT', 'WU', 'WV', 'WW', 'WX', 'WY', 'WZ'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [1, 2, 3]  # Can be 1-3 letters
        },
        'CANADA': {
            'prefixes': ['VE', 'VA', 'VO', 'VY'],
            'format': 'prefix_number_suffix',
            'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            'suffix_length': [2, 3]
        },
        'ENGLAND': {
            'prefixes': ['G', 'M', 'GM', 'GW', 'GI', 'GD', 'GU', 'GJ', '2E', 'M0', 'M1', 'M3', 'M5', 'M6'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'GERMANY': {
            'prefixes': ['DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'JAPAN': {
            'prefixes': ['JA', 'JB', 'JC', 'JD', 'JE', 'JF', 'JG', 'JH', 'JI', 'JJ', 'JK', 'JL', 'JM', 'JN', 'JO', 'JP', 'JQ', 'JR', 'JS'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'ITALY': {
            'prefixes': ['I', 'IK', 'IZ', 'IW', 'IU', 'IQ'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'RUSSIA': {
            'prefixes': ['UA', 'UB', 'UC', 'UD', 'UE', 'UF', 'UG', 'UH', 'UI', 'RA', 'RB', 'RC', 'RD', 'RE', 'RF', 'RG', 'RH', 'RI', 'RJ', 'RK', 'RL', 'RM', 'RN', 'RO', 'RP', 'RQ', 'RR', 'RS', 'RT', 'RU', 'RV', 'RW', 'RX', 'RY', 'RZ'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'SPAIN': {
            'prefixes': ['EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH'],
            'format': 'prefix_number_suffix',
            'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'FRANCE': {
            'prefixes': ['F'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        },
        'BRAZIL': {
            'prefixes': ['PY', 'PP', 'PQ', 'PR', 'PS', 'PT', 'PU', 'PV', 'PW', 'PX', 'ZV', 'ZW', 'ZX', 'ZY', 'ZZ'],
            'format': 'prefix_number_suffix',
            'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            'suffix_length': [2, 3]
        },
        'AUSTRALIA': {
            'prefixes': ['VK'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3, 4]
        },
        'ARGENTINA': {
            'prefixes': ['LU', 'LW', 'AY', 'AZ', 'LO', 'LP', 'LQ', 'LR', 'LS', 'LT', 'LV'],
            'format': 'prefix_number_suffix',
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'suffix_length': [2, 3]
        }
    }
    
    # Special contest/event callsigns
    SPECIAL_FORMATS = {
        'CONTEST_STATION': {
            'prefixes': ['K1', 'W2', 'N3', 'W4', 'K5', 'W6', 'W7', 'W8', 'K9', 'W0'],
            'suffixes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        },
        'SPECIAL_EVENT': {
            'patterns': ['prefix_special_number', 'prefix_number_special'],
            'special_parts': ['FIELD', 'TEST', 'DX', 'CONTEST', 'IOTA', 'SOTA']
        }
    }
    
    def __init__(self):
        """Initialize the callsign generator."""
        self.countries = list(self.COUNTRY_PREFIXES.keys())
        
    def generate_suffix(self, length: int) -> str:
        """Generate a random suffix of specified length."""
        # First character is always a letter
        suffix = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Remaining characters can be letters or sometimes numbers in the last position
        for i in range(1, length):
            if i == length - 1 and random.random() < 0.1:  # 10% chance of number at end
                suffix += random.choice('0123456789')
            else:
                suffix += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
        return suffix
    
    def generate_callsign(self, country: Optional[str] = None, 
                         special_format: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate a random amateur radio callsign.
        
        Args:
            country: Specific country to generate for (None = random)
            special_format: Special format type ('CONTEST_STATION', etc.)
            
        Returns:
            Tuple of (callsign, country)
        """
        if special_format == 'CONTEST_STATION':
            prefix = random.choice(self.SPECIAL_FORMATS['CONTEST_STATION']['prefixes'])
            suffix = random.choice(self.SPECIAL_FORMATS['CONTEST_STATION']['suffixes'])
            return f"{prefix}{suffix}", 'USA'
        
        # Select country
        if country is None:
            # Weight selection towards more common contest countries
            weights = [30, 10, 15, 20, 25, 15, 10, 10, 5, 5, 5, 5]  # Rough contest activity weights
            country = random.choices(self.countries, weights=weights[:len(self.countries)])[0]
        
        country_data = self.COUNTRY_PREFIXES[country]
        
        # Generate callsign
        prefix = random.choice(country_data['prefixes'])
        number = random.choice(country_data['numbers'])
        suffix_length = random.choice(country_data['suffix_length'])
        suffix = self.generate_suffix(suffix_length)
        
        callsign = f"{prefix}{number}{suffix}"
        
        return callsign, country
    
    def generate_portable_callsign(self, base_callsign: str, 
                                  location: Optional[str] = None) -> str:
        """
        Generate a portable/mobile callsign variant.
        
        Args:
            base_callsign: Base callsign
            location: Location indicator (for portable operation)
            
        Returns:
            Modified callsign
        """
        modifiers = []
        
        if location:
            # Portable operation
            modifiers.append(f"{base_callsign}/{location}")
            modifiers.append(f"{location}/{base_callsign}")
        
        # Mobile operation
        modifiers.append(f"{base_callsign}/M")
        modifiers.append(f"{base_callsign}/MM")  # Maritime mobile
        modifiers.append(f"{base_callsign}/AM")  # Aeronautical mobile
        modifiers.append(f"{base_callsign}/P")   # Portable
        modifiers.append(f"{base_callsign}/QRP") # Low power
        
        return random.choice(modifiers)
    
    def generate_batch(self, count: int, countries: Optional[List[str]] = None,
                      include_portable: float = 0.1) -> List[Tuple[str, str]]:
        """
        Generate a batch of callsigns.
        
        Args:
            count: Number of callsigns to generate
            countries: List of countries to use (None = all)
            include_portable: Fraction of portable/mobile callsigns
            
        Returns:
            List of (callsign, country) tuples
        """
        callsigns = []
        
        for _ in range(count):
            if countries:
                country = random.choice(countries)
            else:
                country = None
            
            # Generate base callsign
            if random.random() < 0.05:  # 5% contest stations
                callsign, country = self.generate_callsign(special_format='CONTEST_STATION')
            else:
                callsign, country = self.generate_callsign(country)
            
            # Maybe make it portable/mobile
            if random.random() < include_portable:
                if random.random() < 0.5 and country == 'USA':
                    # US portable with state
                    state = random.choice(['FL', 'CA', 'TX', 'NY', 'PA', 'OH', 'IL', 'MI', 'NC', 'GA'])
                    callsign = self.generate_portable_callsign(callsign, state)
                else:
                    callsign = self.generate_portable_callsign(callsign)
            
            callsigns.append((callsign, country))
        
        return callsigns
    
    def is_valid_callsign(self, callsign: str) -> bool:
        """Check if a string could be a valid amateur radio callsign."""
        # Basic validation
        if not callsign or len(callsign) < 3 or len(callsign) > 10:
            return False
        
        # Must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in callsign)
        has_number = any(c.isdigit() for c in callsign)
        
        # Check for valid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/')
        if not all(c in valid_chars for c in callsign.upper()):
            return False
        
        return has_letter and has_number


def generate_samples():
    """Generate sample callsigns for testing."""
    generator = CallsignGenerator()
    
    print("Random callsigns from various countries:")
    for _ in range(10):
        callsign, country = generator.generate_callsign()
        print(f"  {callsign} ({country})")
    
    print("\nContest station callsigns:")
    for _ in range(5):
        callsign, country = generator.generate_callsign(special_format='CONTEST_STATION')
        print(f"  {callsign}")
    
    print("\nPortable/Mobile callsigns:")
    base_call = "W1ABC"
    print(f"  {generator.generate_portable_callsign(base_call, 'FL')}")
    print(f"  {generator.generate_portable_callsign(base_call)}")


if __name__ == "__main__":
    generate_samples() 